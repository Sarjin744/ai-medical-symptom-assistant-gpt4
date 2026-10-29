"""
Automatic GitHub Publisher for Experiment 6: AI-Powered Medical Symptom Assistant
Course: 24CS2506 – Building Applications using GPT-4
Authorizes via active GitHub CLI / Device Token and uploads project files.
"""

import os
import sys
import re
import base64
import requests

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    hosts_path = r"C:\Users\csarj\AppData\Roaming\GitHub CLI\hosts.yml"
    if not os.path.exists(hosts_path):
        print(f"[-] Error: {hosts_path} not found.")
        sys.exit(1)

    with open(hosts_path, "r", encoding="utf-8") as f:
        text = f.read()

    match = re.search(r"oauth_token:\s*([a-zA-Z0-9_]+)", text)
    if not match:
        print("[-] Error: OAuth token not found in hosts.yml")
        sys.exit(1)

    token = match.group(1)
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }

    # 1. Verify User
    print("[*] Verifying GitHub user credentials...")
    user_res = requests.get("https://api.github.com/user", headers=headers)
    if user_res.status_code != 200:
        print(f"[-] Auth failed: {user_res.status_code} - {user_res.text}")
        sys.exit(1)

    username = user_res.json()["login"]
    repo_name = "ai-medical-symptom-assistant-gpt4"
    print(f"[✓] Authenticated as: {username}")

    # 2. Check / Create Repository
    repo_url = f"https://api.github.com/repos/{username}/{repo_name}"
    check = requests.get(repo_url, headers=headers)

    if check.status_code == 404:
        print(f"[*] Creating GitHub repository '{repo_name}'...")
        create_payload = {
            "name": repo_name,
            "description": "Course: 24CS2506 | Experiment 6: AI-Powered Medical Symptom Assistant using GPT-4 / Clinical NLP Pipeline",
            "private": False,
            "auto_init": True
        }
        cr = requests.post("https://api.github.com/user/repos", headers=headers, json=create_payload)
        if cr.status_code in (200, 201):
            print(f"[✓] Repository created: https://github.com/{username}/{repo_name}")
        else:
            print(f"[-] Error creating repo: {cr.status_code} - {cr.text}")
            sys.exit(1)
    else:
        print(f"[✓] Target repository exists: https://github.com/{username}/{repo_name}")

    # 3. Files to upload
    include_files = [
        "README.md",
        "app.py",
        "cli.py",
        "test_suite.py",
        "run_app.bat",
        "run_cli.bat",
        ".gitignore",
        "ex6.pdf",
        "upload_to_github.py",
        "data/sample_cases.json",
        "engine/__init__.py",
        "engine/models.py",
        "engine/symptom_extractor.py",
        "engine/context_analyzer.py",
        "engine/prompt_builder.py",
        "engine/llm_client.py",
        "engine/assistant.py",
        "saved_reports/case_lab_manual_20261030_014004.txt",
        "saved_reports/case_lab_manual_20261030_014004.json",
        "saved_reports/case_lab_manual_20261030_014004.md",
        "saved_reports/case_emergency_cardiac_20261030_014014.txt",
        "saved_reports/case_emergency_cardiac_20261030_014014.json",
        "saved_reports/case_emergency_cardiac_20261030_014014.md"
    ]

    print(f"\n[*] Uploading {len(include_files)} files to {username}/{repo_name}...")
    uploaded_count = 0

    for rel_path in include_files:
        norm_rel = rel_path.replace("\\", "/")
        if not os.path.exists(norm_rel):
            print(f"[-] Skipping missing: {norm_rel}")
            continue

        with open(norm_rel, "rb") as f:
            content_bytes = f.read()

        b64_content = base64.b64encode(content_bytes).decode("utf-8")
        file_url = f"https://api.github.com/repos/{username}/{repo_name}/contents/{norm_rel}"

        # Check existing file SHA for updates
        file_check = requests.get(file_url, headers=headers)
        sha = file_check.json().get("sha") if file_check.status_code == 200 else None

        commit_message = f"Add {norm_rel} for Experiment 6" if not sha else f"Update {norm_rel}"
        payload = {
            "message": commit_message,
            "content": b64_content
        }
        if sha:
            payload["sha"] = sha

        put_res = requests.put(file_url, headers=headers, json=payload)
        if put_res.status_code in (200, 201):
            print(f"  [+] Uploaded: {norm_rel}")
            uploaded_count += 1
        else:
            print(f"  [-] Failed: {norm_rel} ({put_res.status_code}) - {put_res.text}")

    repo_html_url = f"https://github.com/{username}/{repo_name}"
    print("\n" + "=" * 65)
    print(" 🎉 EXPERIMENT 6 SUCCESSFULLY PUBLISHED TO GITHUB!")
    print(f" Repository URL: {repo_html_url}")
    print(f" Total Files Uploaded: {uploaded_count}/{len(include_files)}")
    print("=" * 65)


if __name__ == "__main__":
    main()
