import ollama
try:
    models = ollama.list()
    print("Full result:", models)
    if hasattr(models, 'models'):
        print("Models attribute:", models.models)
        for m in models.models:
            print("Model object:", m)
            print("Model keys/attributes:", dir(m))
    elif isinstance(models, dict):
        print("Models key:", models.get('models'))
except Exception as e:
    print("Error:", e)
