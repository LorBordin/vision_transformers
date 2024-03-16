def print_model_stats(model, args):

    n_params = sum(p.numel() for p in model.parameters())

    print()
    print("*** ----- Model Stats ----- ***")
    print(f"[INFO] Input dimension: {args['input_size']}.")
    print(f"[INFO] Number of splits per size: {args['n_patches']}.")
    print(f"[INFO] Number of transformer blocks: {args['n_blocks']}.")
    print(f"[INFO] Embbedding dimension: {args['emb_size']}.")
    print(f"[INFO] Number of attention heads per block: {args['n_heads']}.")
    print(f"[INFO] Number of classification classes: {args['out_dim']}.")    
    print("*** _______________________ ***")
    print("[INFO] Number of training parameters:", n_params, end="\n")
    print("*** ----------------------- ***")
    print()