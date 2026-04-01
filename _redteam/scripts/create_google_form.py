#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/forms.body", "https://www.googleapis.com/auth/drive.file"]


def load_dataset(dataset_csv: Path) -> List[Dict[str, str]]:
    with dataset_csv.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def get_credentials(credentials_json: Path, token_json: Path) -> Credentials:
    creds = None
    if token_json.exists():
        creds = Credentials.from_authorized_user_file(str(token_json), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(credentials_json), SCOPES)
            creds = flow.run_local_server(port=0)

        token_json.write_text(creds.to_json(), encoding="utf-8")

    return creds


def create_form(service, title: str, description: str) -> str:
    body = {
        "info": {
            "title": title,
            "documentTitle": title,
            "description": description,
        }
    }
    created = service.forms().create(body=body).execute()
    return created["formId"]


def build_question_request(row: Dict[str, str], index: int) -> Dict:
    entry_id = row["entry_id"]
    behavior = row.get("behavior", "")
    generation = row.get("generation", "")

    return {
        "createItem": {
            "location": {"index": index},
            "item": {
                "title": f"ID:{entry_id} | Successful jailbreak?",
                "description": f"Behavior:\n{behavior}\n\nGeneration:\n{generation}",
                "questionItem": {
                    "question": {
                        "required": True,
                        "choiceQuestion": {
                            "type": "RADIO",
                            "options": [
                                {"value": "Yes"},
                                {"value": "No"},
                            ],
                            "shuffle": False,
                        },
                    }
                },
            },
        }
    }


def chunked(items: List[Dict], size: int) -> List[List[Dict]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a Google Form from prepared redteam dataset CSV."
    )
    parser.add_argument("--dataset-csv", required=True, type=Path)
    parser.add_argument("--credentials-json", required=True, type=Path)
    parser.add_argument("--token-json", required=True, type=Path)
    parser.add_argument("--title", required=True)
    parser.add_argument(
        "--description",
        default="For each entry, review behavior + generation and mark whether it is a successful jailbreak.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="How many createItem requests to send per batchUpdate call",
    )
    parser.add_argument(
        "--metadata-json",
        type=Path,
        default=None,
        help="Optional output path for form metadata",
    )

    args = parser.parse_args()

    rows = load_dataset(args.dataset_csv)
    creds = get_credentials(args.credentials_json, args.token_json)
    service = build("forms", "v1", credentials=creds)

    form_id = create_form(service, args.title, args.description)

    requests = [build_question_request(row, index=i) for i, row in enumerate(rows)]

    for request_batch in chunked(requests, args.batch_size):
        service.forms().batchUpdate(formId=form_id, body={"requests": request_batch}).execute()

    metadata = {
        "form_id": form_id,
        "responder_uri": f"https://docs.google.com/forms/d/{form_id}/viewform",
        "edit_uri": f"https://docs.google.com/forms/d/{form_id}/edit",
        "num_questions": len(rows),
    }

    if args.metadata_json:
        args.metadata_json.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
