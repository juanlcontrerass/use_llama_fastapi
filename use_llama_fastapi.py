from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import uvicorn
import os
import tempfile
import shutil
from dotenv import load_dotenv

# Import GCP storage only if needed
try:
    from google.cloud import storage
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False
    print("Advertencia: google-cloud-storage no está instalado. Solo modo LOCAL disponible.")

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Text Generation API",
    description="API para generación de texto usando modelo entrenado desde local o GCP",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration - Environment variables
USE_GCP = os.getenv("USE_GCP", "false").lower() == "true"
GCP_BUCKET_NAME = os.getenv("GCP_BUCKET_NAME", "")
GCP_MODEL_PATH = os.getenv("GCP_MODEL_PATH", "trained_model/")
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "trained_model")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")

# Global variables for model and pipeline
model = None
tokenizer = None
text_generation = None
device_info = None
model_source = None
temp_model_dir = None

# Request model
class GenerationRequest(BaseModel):
    prompt: str = Field(..., description="El texto de entrada para generar la respuesta")
    max_new_tokens: int = Field(75, ge=1, le=500, description="Número máximo de tokens a generar")
    temperature: float = Field(0.7, ge=0.1, le=2.0, description="Temperatura para el muestreo")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Parámetro top-p para nucleus sampling")
    do_sample: bool = Field(True, description="Si se debe usar muestreo")

# Response model
class GenerationResponse(BaseModel):
    prompt: str
    generated_text: str
    model_info: dict

def download_model_from_gcp(bucket_name: str, model_path: str, local_dir: str) -> Path:
    """
    Download model from GCP bucket to local directory
    
    Args:
        bucket_name: Name of the GCP bucket
        model_path: Path to model inside the bucket (folder)
        local_dir: Local directory to download the model
    
    Returns:
        Path to the downloaded model
    """
    if not GCP_AVAILABLE:
        raise Exception("google-cloud-storage no está instalado. Ejecuta: pip install google-cloud-storage")
    
    try:
        print(f"Iniciando descarga del modelo desde GCS: gs://{bucket_name}/{model_path}")
        
        # Initialize GCS client
        if GOOGLE_APPLICATION_CREDENTIALS:
            storage_client = storage.Client.from_service_account_json(GOOGLE_APPLICATION_CREDENTIALS)
        else:
            # Use default credentials
            storage_client = storage.Client()
        
        bucket = storage_client.bucket(bucket_name)
        
        # Create local directory
        local_model_path = Path(local_dir)
        local_model_path.mkdir(parents=True, exist_ok=True)
        
        # List all blobs with the prefix
        blobs = bucket.list_blobs(prefix=model_path)
        
        downloaded_files = 0
        blob_list = list(blobs)
        
        if not blob_list:
            raise Exception(f"No se encontraron archivos en gs://{bucket_name}/{model_path}")
        
        print(f"Encontrados {len(blob_list)} archivos en el bucket")
        
        for blob in blob_list:
            # Skip if it's just a folder marker
            if blob.name.endswith('/'):
                continue
            
            # Get relative path
            relative_path = blob.name[len(model_path):].lstrip('/')
            if not relative_path:
                continue
            
            # Create local file path
            local_file_path = local_model_path / relative_path
            local_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Download the file
            print(f"  Descargando: {blob.name} ({blob.size / (1024*1024):.2f} MB)")
            blob.download_to_filename(str(local_file_path))
            downloaded_files += 1
        
        print(f"✓ Descarga completada: {downloaded_files} archivos descargados")
        return local_model_path
        
    except Exception as e:
        raise Exception(f"Error al descargar modelo desde GCP: {str(e)}")

def load_model_local(model_path: str) -> tuple:
    """Load model from local path"""
    try:
        path = Path(model_path).resolve()
        print(f"Cargando modelo desde ruta local: {path}")
        
        if not path.exists():
            raise FileNotFoundError(f"La ruta del modelo no existe: {path}")
        
        # Verify required files exist
        required_files = ["config.json", "pytorch_model.bin"]
        missing_files = [f for f in required_files if not (path / f).exists() and not any((path / f"pytorch_model-{i:05d}-of-*.bin").exists() for i in range(1, 100))]
        
        print(f"  Verificando archivos del modelo...")
        
        model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        
        print(f"✓ Modelo cargado exitosamente desde local")
        return model, tokenizer, f"local:{path}"
        
    except Exception as e:
        raise Exception(f"Error al cargar modelo desde local: {str(e)}")

def load_model_gcp(bucket_name: str, model_path: str) -> tuple:
    """Load model from GCP bucket"""
    global temp_model_dir
    
    try:
        # Create a temporary directory for the model
        temp_model_dir = tempfile.mkdtemp(prefix="gcp_model_")
        print(f"Directorio temporal creado: {temp_model_dir}")
        
        # Download model from GCP
        local_path = download_model_from_gcp(bucket_name, model_path, temp_model_dir)
        
        # Load the model
        print(f"Cargando modelo desde directorio descargado...")
        model = AutoModelForCausalLM.from_pretrained(local_path, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True)
        
        print(f"✓ Modelo cargado exitosamente desde GCP")
        return model, tokenizer, f"gcp:gs://{bucket_name}/{model_path}"
        
    except Exception as e:
        # Clean up temp directory if there's an error
        if temp_model_dir and os.path.exists(temp_model_dir):
            shutil.rmtree(temp_model_dir)
        raise Exception(f"Error al cargar modelo desde GCP: {str(e)}")

@app.on_event("startup")
async def load_model():
    """Load model and tokenizer on startup"""
    global model, tokenizer, text_generation, device_info, model_source
    
    try:
        print("\n" + "="*80)
        print("INICIANDO CARGA DEL MODELO")
        print("="*80)
        print(f"Configuración:")
        print(f"  - USE_GCP: {USE_GCP}")
        print(f"  - GCP_BUCKET_NAME: {GCP_BUCKET_NAME if USE_GCP else 'N/A'}")
        print(f"  - GCP_MODEL_PATH: {GCP_MODEL_PATH if USE_GCP else 'N/A'}")
        print(f"  - LOCAL_MODEL_PATH: {LOCAL_MODEL_PATH if not USE_GCP else 'N/A'}")
        print(f"  - CREDENTIALS: {'Configuradas' if GOOGLE_APPLICATION_CREDENTIALS else 'Default'}")
        print("-"*80)
        
        # Load model based on configuration
        if USE_GCP:
            if not GCP_AVAILABLE:
                raise Exception(
                    "No se puede usar modo GCP sin google-cloud-storage instalado.\n"
                    "Instala con: pip install google-cloud-storage\n"
                    "O cambia USE_GCP=false en tu archivo .env"
                )
            
            if not GCP_BUCKET_NAME:
                raise ValueError("GCP_BUCKET_NAME no está configurado. Revisa tu archivo .env")
            
            print("Modo: CARGA DESDE GCP")
            model, tokenizer, model_source = load_model_gcp(GCP_BUCKET_NAME, GCP_MODEL_PATH)
        else:
            print("Modo: CARGA LOCAL")
            model, tokenizer, model_source = load_model_local(LOCAL_MODEL_PATH)
        
        # Check if GPU is available and move the model to GPU
        device = 0 if torch.cuda.is_available() else -1
        device_info = {
            'device': 'GPU' if device >= 0 else 'CPU',
            'cuda_available': torch.cuda.is_available(),
            'model_source': model_source
        }
        
        if torch.cuda.is_available():
            print(f"  GPU detectada: {torch.cuda.get_device_name(0)}")
        
        print(f"\nDispositivo de cómputo: {device_info['device']}")
        print(f"Origen del modelo: {model_source}")
        
        # Create text generation pipeline
        print("\nCreando pipeline de generación de texto...")
        text_generation = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=device
        )
        
        print("="*80)
        print("✓ MODELO CARGADO Y LISTO PARA USO")
        print("="*80 + "\n")
        
    except Exception as e:
        print("="*80)
        print(f"✗ ERROR AL CARGAR MODELO")
        print("="*80)
        print(f"Error: {str(e)}\n")
        raise e

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global temp_model_dir
    
    print("\nCerrando servidor...")
    
    # Clean up temporary files if using GCP
    if USE_GCP and temp_model_dir and os.path.exists(temp_model_dir):
        print(f"Limpiando directorio temporal: {temp_model_dir}")
        try:
            shutil.rmtree(temp_model_dir)
            print("✓ Limpieza completada")
        except Exception as e:
            print(f"Advertencia: No se pudo eliminar directorio temporal: {e}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Text Generation API",
        "status": "running",
        "version": "2.0.0",
        "configuration": {
            "use_gcp": USE_GCP,
            "model_source": model_source
        },
        "endpoints": {
            "health": "/health - Verificar estado del servicio",
            "config": "/config - Ver configuración actual",
            "generate": "/generate (POST) - Generar texto individual",
            "batch-generate": "/batch-generate (POST) - Generar texto en lote",
            "docs": "/docs - Documentación interactiva"
        }
    }

@app.get("/config")
async def get_config():
    """Get current configuration"""
    return {
        "mode": "GCP" if USE_GCP else "LOCAL",
        "use_gcp": USE_GCP,
        "gcp_config": {
            "bucket_name": GCP_BUCKET_NAME if USE_GCP else "N/A",
            "model_path": GCP_MODEL_PATH if USE_GCP else "N/A",
            "credentials_configured": bool(GOOGLE_APPLICATION_CREDENTIALS)
        } if USE_GCP else None,
        "local_config": {
            "model_path": LOCAL_MODEL_PATH
        } if not USE_GCP else None,
        "model_source": model_source,
        "model_loaded": text_generation is not None
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if text_generation is None:
        raise HTTPException(
            status_code=503, 
            detail="Modelo no cargado. El servidor puede estar iniciando."
        )
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "device": device_info,
        "model_source": model_source,
        "mode": "GCP" if USE_GCP else "LOCAL"
    }

@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    """Generate text based on the provided prompt"""
    
    if text_generation is None:
        raise HTTPException(
            status_code=503, 
            detail="Modelo no disponible aún. Espera a que termine de cargar."
        )
    
    try:
        # Generate text
        result = text_generation(
            request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=request.do_sample
        )[0]["generated_text"]
        
        return GenerationResponse(
            prompt=request.prompt,
            generated_text=result,
            model_info={
                "max_new_tokens": request.max_new_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "device": device_info['device'],
                "model_source": model_source
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error al generar texto: {str(e)}"
        )

@app.post("/batch-generate")
async def batch_generate(
    prompts: list[str], 
    max_new_tokens: int = 75, 
    temperature: float = 0.7, 
    top_p: float = 0.9
):
    """Generate text for multiple prompts (max 10)"""
    
    if text_generation is None:
        raise HTTPException(
            status_code=503, 
            detail="Modelo no disponible aún"
        )
    
    if len(prompts) > 10:
        raise HTTPException(
            status_code=400, 
            detail="Máximo 10 prompts permitidos por lote"
        )
    
    if not prompts:
        raise HTTPException(
            status_code=400,
            detail="Debes proporcionar al menos un prompt"
        )
    
    try:
        results = []
        for i, prompt in enumerate(prompts, 1):
            print(f"Procesando prompt {i}/{len(prompts)}")
            result = text_generation(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True
            )[0]["generated_text"]
            
            results.append({
                "prompt": prompt,
                "generated_text": result
            })
        
        return {
            "results": results,
            "count": len(results),
            "model_info": {
                "device": device_info['device'],
                "model_source": model_source,
                "parameters": {
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_p": top_p
                }
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error en generación por lote: {str(e)}"
        )

if __name__ == "__main__":
    # Run the server
    print("\n🚀 Iniciando servidor FastAPI...")
    print(f"📍 URL: http://localhost:8000")
    print(f"📚 Docs: http://localhost:8000/docs\n")
    
    # Windows compatibility fix
    import sys
    if sys.platform == "win32":
        import multiprocessing
        multiprocessing.freeze_support()
    
    uvicorn.run(
        app,  # Pass app directly instead of string
        host="0.0.0.0",
        port=8000,
        reload=False
    )