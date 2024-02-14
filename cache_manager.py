def get_cache():
    g = globals()
    if "_multimodal_rag" not in g:
        g["_multimodal_rag"] = {}

    return g["_multimodal_rag"]
