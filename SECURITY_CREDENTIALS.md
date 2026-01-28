# Security: Handling Credentials

## ⚠️ IMPORTANT: Credentials Management

This document outlines how to properly handle sensitive credentials in this project.

### What Happened?
A Google Cloud service account credentials file (`noble-district-459911-t9-1c0ed90c64dd.json`) containing a private key was accidentally committed to the repository history. This file has been removed from:
- The working directory
- The entire git history using `git-filter-repo`

### Action Required
**The exposed service account key MUST be revoked immediately:**
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Navigate to IAM & Admin > Service Accounts
3. Find the service account: `bigquery-data-access@noble-district-459911-t9.iam.gserviceaccount.com`
4. Delete the key with ID: `1c0ed90c64dd7e8c4452673eb0c260e28f65d85f`
5. Generate a new key if needed

### Best Practices for Credentials

#### Never Commit Credentials
- **NEVER** commit API keys, passwords, tokens, or service account files to git
- Use `.gitignore` to prevent accidental commits (already configured)
- Store credentials locally outside the repository

#### How to Use Credentials Properly

1. **Store credentials outside the repository:**
   ```bash
   # Store in your home directory or a secure location
   mkdir -p ~/.gcloud
   mv noble-district-459911-t9-1c0ed90c64dd.json ~/.gcloud/
   chmod 600 ~/.gcloud/noble-district-459911-t9-1c0ed90c64dd.json
   ```

2. **Reference credentials using environment variables:**
   ```python
   import os
   os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.expanduser('~/.gcloud/noble-district-459911-t9-1c0ed90c64dd.json')
   ```

3. **For production, use secret management services:**
   - GitHub Secrets for CI/CD
   - Google Secret Manager
   - HashiCorp Vault
   - AWS Secrets Manager

#### What's Protected in .gitignore
The `.gitignore` file now includes patterns to prevent credential files:
```
# Google Cloud credentials - NEVER commit these files
*.json
!pyproject.toml
*credentials*.json
*service-account*.json
```

### Verification
To verify credentials are not in the repository:
```bash
# Check current files
git ls-files | grep -i credential
git ls-files | grep -i "\.json$" | grep -v pyproject.toml

# Check history
git log --all --full-history -- "*credentials*.json"
git log --all --full-history -- "*service-account*.json"
```

### If You Accidentally Commit Credentials
1. **STOP** - Don't push if you haven't already
2. **Revoke/Rotate** the credentials immediately
3. Remove from git history:
   ```bash
   pip install git-filter-repo
   git-filter-repo --path path/to/credential-file.json --invert-paths --force
   ```
4. Force push (requires coordination with team):
   ```bash
   git push --force-with-lease origin branch-name
   ```

### Additional Resources
- [GitHub: Removing sensitive data](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository)
- [Git Filter-Repo](https://github.com/newren/git-filter-repo)
- [Google Cloud: Best practices for managing service account keys](https://cloud.google.com/iam/docs/best-practices-for-managing-service-account-keys)
