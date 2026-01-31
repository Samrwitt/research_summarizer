"""
Model configurations for experimentation.
Each model has specific strengths and trade-offs.
"""

MODEL_REGISTRY = {
    # Current production model
    "led-base": {
        "model_id": "allenai/led-base-16384",
        "max_input_length": 16384,
        "description": "Longformer Encoder-Decoder for long documents (production)",
        "gpu_memory_gb": 6,
        "speed": "slow",
        "quality": "high",
        "domain": "scientific (arXiv/PubMed)"
    },
    
    # Faster alternative (current fallback)
    "distilbart": {
        "model_id": "sshleifer/distilbart-cnn-12-6",
        "max_input_length": 1024,
        "description": "Distilled BART - fast but limited context",
        "gpu_memory_gb": 2,
        "speed": "fast",
        "quality": "medium",
        "domain": "news/general"
    },
    
    # Experimental models to try
    "led-large": {
        "model_id": "allenai/led-large-16384",
        "max_input_length": 16384,
        "description": "Larger LED - better quality, slower",
        "gpu_memory_gb": 12,
        "speed": "very_slow",
        "quality": "very_high",
        "domain": "scientific"
    },
    
    "longt5-base": {
        "model_id": "google/long-t5-tglobal-base",
        "max_input_length": 16384,
        "description": "Google's Long-T5 with transient global attention",
        "gpu_memory_gb": 5,
        "speed": "medium",
        "quality": "high",
        "domain": "general"
    },
    
    "longt5-large": {
        "model_id": "google/long-t5-tglobal-large",
        "max_input_length": 16384,
        "description": "Larger Long-T5 variant",
        "gpu_memory_gb": 10,
        "speed": "slow",
        "quality": "very_high",
        "domain": "general"
    },
    
    "bart-large-cnn": {
        "model_id": "facebook/bart-large-cnn",
        "max_input_length": 1024,
        "description": "Original BART - good for shorter texts",
        "gpu_memory_gb": 4,
        "speed": "medium",
        "quality": "high",
        "domain": "news/general"
    },
    
    "pegasus-large": {
        "model_id": "google/pegasus-large",
        "max_input_length": 1024,
        "description": "PEGASUS - pre-trained with gap-sentence generation",
        "gpu_memory_gb": 5,
        "speed": "medium",
        "quality": "high",
        "domain": "news/general"
    },
    
    "flan-t5-base": {
        "model_id": "google/flan-t5-base",
        "max_input_length": 512,
        "description": "Instruction-tuned T5 - versatile but short context",
        "gpu_memory_gb": 3,
        "speed": "fast",
        "quality": "medium",
        "domain": "general"
    },
    
    "flan-t5-large": {
        "model_id": "google/flan-t5-large",
        "max_input_length": 512,
        "description": "Larger instruction-tuned T5",
        "gpu_memory_gb": 6,
        "speed": "medium",
        "quality": "high",
        "domain": "general"
    }
}

def get_model_config(model_key):
    """Get configuration for a specific model."""
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_key]

def list_models():
    """Print all available models with their specs."""
    print("\n" + "="*80)
    print("AVAILABLE MODELS FOR EXPERIMENTATION")
    print("="*80)
    
    for key, config in MODEL_REGISTRY.items():
        print(f"\n🔹 {key}")
        print(f"   Model ID: {config['model_id']}")
        print(f"   Max Context: {config['max_input_length']} tokens")
        print(f"   GPU Memory: {config['gpu_memory_gb']} GB")
        print(f"   Speed: {config['speed']} | Quality: {config['quality']}")
        print(f"   Domain: {config['domain']}")
        print(f"   Description: {config['description']}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    list_models()
