# ValidReuse Method Explorer

Streamlit app for comparing q10(LRV) evaluation methods on the same editable integer input table:

- Empirical bootstrap
- Bayesian approximation
- Bayesian (PyMC)

## Use the local Conda environment

```zsh
conda activate validreuse
```

## Install dependencies

```zsh
pip install -r requirements.txt
```

Recommended:
- Use Python 3.13 for the full app, including PyMC.
- The older `validreuse/` Python 3.14 environment is still present, but PyMC is not reliable there.

## Run the app

```zsh
streamlit run app.py
```

## Deploy With Docker

Build the image:

```zsh
docker build -t validreuse-app .
```

Run it locally:

```zsh
docker run --rm -p 8501:8501 validreuse-app
```

Then open `http://localhost:8501`.

## Deploy On A Server

1. Copy the project to the server.
2. Install Docker.
3. Build the image:

```zsh
docker build -t validreuse-app .
```

4. Run the container in the background:

```zsh
docker run -d --name validreuse -p 8501:8501 --restart unless-stopped validreuse-app
```

5. Put Nginx or Caddy in front of port `8501` if you want HTTPS and a public domain.

## Streamlit Cloud

This app can also be deployed from a GitHub repo on Streamlit Community Cloud with `app.py` as the entrypoint.

Note:
- PyMC makes deployment heavier than a simple Streamlit app.
- If you want the lightest deployment, we can make the PyMC method optional or disable it in cloud builds.
