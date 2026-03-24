# GitHub Repository Setup & Sync

## Repository
- **Remote**: https://github.com/darshpatel1052/CV-Project

---

## Option A: Clone into the Project directory (recommended for empty directory)

Since the directory is currently empty, initialize git and pull from the remote:

```bash
# Navigate to the project directory
cd /home/predator-linux/Acads/Sem6/CV/Project

# Initialize a new git repository
git init

# Add the remote origin
git remote add origin https://github.com/darshpatel1052/CV-Project.git

# Fetch all branches and history
git fetch origin

# Checkout the main branch (use 'master' if 'main' doesn't exist)
git checkout -b main origin/main
# OR if the default branch is 'master':
# git checkout -b master origin/master
```

## Option B: Clone directly (alternative)

If you prefer to clone fresh (this creates a `CV-Project` subfolder, so you'd need to move files):

```bash
# Clone into a temp folder
git clone https://github.com/darshpatel1052/CV-Project.git /tmp/CV-Project-clone

# Move everything (including .git) into the project directory
cp -a /tmp/CV-Project-clone/. /home/predator-linux/Acads/Sem6/CV/Project/

# Clean up
rm -rf /tmp/CV-Project-clone
```

---

## Syncing Changes

### Pull latest changes from GitHub
```bash
cd /home/predator-linux/Acads/Sem6/CV/Project
git pull origin main
```

### Push local changes to GitHub
```bash
cd /home/predator-linux/Acads/Sem6/CV/Project

# Stage all changes
git add .

# Commit with a message
git commit -m "Your commit message here"

# Push to remote
git push origin main
```

### Check sync status
```bash
cd /home/predator-linux/Acads/Sem6/CV/Project

# Check current status
git status

# Check remote info
git remote -v

# View recent commits
git log --oneline -5
```

---

## SSH Setup (Optional, for password-free push/pull)

If you want to use SSH instead of HTTPS:

```bash
# Generate SSH key (skip if you already have one)
ssh-keygen -t ed25519 -C "your_email@example.com"

# Copy the public key
cat ~/.ssh/id_ed25519.pub
# → Add this key to GitHub: Settings → SSH and GPG keys → New SSH key

# Switch remote to SSH
cd /home/predator-linux/Acads/Sem6/CV/Project
git remote set-url origin git@github.com:darshpatel1052/CV-Project.git
```

---

## Git Identity Setup (if not already configured)

```bash
git config --global user.name "Your Name"
git config --global user.email "your_email@example.com"
```
