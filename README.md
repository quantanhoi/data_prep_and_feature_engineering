# Project Setup and Deployment Guide

## Prerequisites

- Python 3.12 installed  
- Poetry package manager installed  
- Access to Confluence with API credentials  

## Environment Setup

### 1. Configure Poetry Environment

Set the Python version for your Poetry environment:

```bash
poetry env use python3.12
```

### 2. Activate Poetry Environment

Activate the Poetry-managed virtual environment:

```bash
poetry env activate
```

### 3. Alternative Activation Methods

If the above doesn’t work, you can manually activate using the full path:

```bash
source /home/trungthieu/.cache/pypoetry/virtualenvs/first-test-demo-hQf9xYdT-py3.12/bin/activate
```

Or use Poetry shell:

```bash
poetry shell
```
if you use shell to activate environment, use ```exit``` to deactivate environment

Or activate a local virtual environment:

```bash
source .venv/bin/activate
```

### Poetry install
If you have a problem with poetry.lock, try this: 

```bash
sudo apt install python3-pip
pip install --upgrade poetry
```
then activate the environment and install:
```bash
poetry install
```


## Deployment

### Publish to Confluence

Once your environment is activated, deploy your documentation to Confluence:

```bash
sudo ./dtcw publishToConfluence \
  -PconfluenceUser=thieuquangtrung1999@gmail.com \
  -PconfluencePass=<apikey>
```

**Important:** Replace `<apikey>` with your actual Confluence API token.

## Notes

- The project uses Poetry for dependency management and virtual-environment isolation.  
- Multiple activation methods are provided for different scenarios.  
- Confluence deployment requires proper API credentials.  
- The `dtcw` script appears to be a custom deployment tool for this project.  

## Troubleshooting

- If Poetry environment activation fails, try using the manual activation path.  
- Ensure your Confluence API key has proper permissions for publishing.  
- Verify Python 3.12 is properly installed and accessible to Poetry.  
