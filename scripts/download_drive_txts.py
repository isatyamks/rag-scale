import os
import argparse
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
import io

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


def get_creds():
    creds = None
    if os.path.exists("credentials.json"):
        flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
        creds = flow.run_local_server(port=8080)
    else:
        raise FileNotFoundError("credentials.json not found.")
    return creds


def download_txt_files(service, folder_id, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    query = f"'{folder_id}' in parents and mimeType='text/plain' and trashed=false"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get("files", [])
    if not files:
        print("No .txt files found in the folder.")
        return
    for file in files:
        file_id = file["id"]
        file_name = file["name"]
        output_path = os.path.join(output_dir, file_name)
        request = service.files().get_media(fileId=file_id)
        fh = io.FileIO(output_path, "wb")
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        print(f"Downloaded {file_name} to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Download .txt files from Google Drive folder.")
    parser.add_argument("--folder-id", required=True, help="Google Drive folder ID")
    parser.add_argument("--output-dir", required=True, help="Directory to save .txt files")
    args = parser.parse_args()

    creds = get_creds()
    service = build("drive", "v3", credentials=creds)
    download_txt_files(service, args.folder_id, args.output_dir)


if __name__ == "__main__":
    main()
