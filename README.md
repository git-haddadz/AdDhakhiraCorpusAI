# AdDhakhiraCorpusAI - Modal Deployment Guide

This branch deploys AdDhakhiraCorpusAI on Modal as an online Gradio web app for the default local LLM pipeline.

## 1) Clone the repository

```bash
git clone --branch deploy/modal https://github.com/git-haddadz/AdDhakhiraCorpusAI.git
cd AdDhakhiraCorpusAI
```

## 2) Create a Modal account

Create a [Modal](https://modal.com/) account and add a payment method.

You can also connect Modal to your GitHub account, but the deployment below is done from your local terminal with the [Modal CLI](https://modal.com/docs/reference/cli).

## 3) Install Modal on your local machine

Install the [Modal CLI](https://modal.com/docs/guide) locally:

```bash
pip install modal
```

If your machine needs Python to be called explicitly:

```bash
python -m pip install modal
```

## 4) Log in to Modal

From the repository root:

```bash
modal setup
```

If `modal setup` does not work, try:

```bash
python -m modal setup
```

Follow the browser login flow.

Check that Modal is connected:

```bash
modal profile current
```

## 5) Choose the GPU

Open `modal_app.py` and choose the GPU:

```python
GPU = "NAME_GPU"
```

Recommended:

```python
GPU = "A100-80GB"
```

You may choose another Modal GPU if your account supports it, but `A100-80GB` is the recommended option for the default models used by this branch. See Modal GPU pricing here: https://modal.com/pricing

## 6) Prepare the Modal volume

Run this once before the first deployment:

```bash
modal run modal_app.py --prepare
```

This downloads the default models into the Modal volume and prepares the dense FAISS index.

## 7) Deploy the web app

After the volume is ready:

```bash
modal deploy modal_app.py
```

Modal prints the web URL at the end of the deployment:

```text
https://<your-modal-workspace>--addhakhira-webapp.modal.run
```

Open the URL in your browser.

## 8) Use the web app

Recommended interface settings:

- Backend: `Default`
- Retrieval dense: checked

You can write the question in Arabic or French.

When the answer is ready, use the download button to save the generated bibliographic synthesis.

## 9) Redeploy after code changes

If you change the Python code, the UI, the Dockerfile, or `modal_app.py`, deploy again:

```bash
modal deploy modal_app.py
```

You usually do not need to run `modal run modal_app.py --prepare` again unless you changed the models or need to rebuild the dense index.
