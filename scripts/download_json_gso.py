import google.auth
from google.cloud import storage
import json
import os

def generate_gso_urls():
    """
    Connects to GCS by initiating a user login flow directly from Python,
    bypassing the need for the gcloud SDK. This is the most direct and
    user-friendly method for this task.
    """
    try:
       
        print("Attempting to authenticate with Google...")
        credentials, project = google.auth.default(
            scopes=["https://www.googleapis.com/auth/devstorage.read_only"]
        )
        print("Authentication successful.")

       
        storage_client = storage.Client(credentials=credentials)
        
        bucket_name = "gso"
        
        print(f"Connecting to Google Cloud bucket '{bucket_name}'...")
        
        blobs = storage_client.list_blobs(bucket_name)
        
        print("Successfully connected. Iterating through files to find all .glb objects...")
        
        base_url = f"https://storage.googleapis.com/{bucket_name}/"
        url_list = []
        count = 0
        for blob in blobs:
            if blob.name.endswith(".glb"):
                url_list.append(f"{base_url}{blob.name}")
                count += 1
                if count % 100 == 0:
                    print(f"Found {count} .glb files...", end='\r')

        print(f"\nIteration complete. Total .glb files found: {count}")

        if not url_list:
            print("\nERROR: Found 0 .glb files.")
            return

        output_filename = "gso_urls_final.json"
        with open(output_filename, "w") as f:
            json.dump(url_list, f, indent=2)

        print(f"\nSuccessfully created {output_filename}")

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        print("\nPlease ensure you have run 'pip install google-cloud-storage google-auth'.")

if __name__ == "__main__":
    generate_gso_urls()