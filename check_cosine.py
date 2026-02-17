import torch
import glob

for exp_name in ['synced_transfer_init-mnist_mlp', 'synced_transfer_init-mnist_mlp_random']:
    files = glob.glob(f'__outputs__/{exp_name}/*.synced_transfer_results.pth')
    if files:
        ckp = torch.load(files[0], map_location='cpu')
        print(f"\n{exp_name}:")
        print(f"  Keys: {list(ckp.keys())}")
        if 'cosine_similarities' in ckp:
            print(f"  Cosine similarities count: {len(ckp['cosine_similarities'])}")
            if ckp['cosine_similarities']:
                print(f"  First 3 entries: {ckp['cosine_similarities'][:3]}")
                print(f"  Last 3 entries: {ckp['cosine_similarities'][-3:]}")
        else:
            print("  NO cosine_similarities found!")
    else:
        print(f"No files found for {exp_name}")
