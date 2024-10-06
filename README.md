[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/AKNyfGde)

You can test my model deployment at : 

[Version 1] https://carpriceprediction-e8c8u4opn2xr6m93gi78ef.streamlit.app/
[Version 2] https://python-st124973.ml.brain.cs.ait.ac.th/ 

If you want to run in your local, just download Docker [here](https://www.docker.com/get-started/) and type

- docker build -t my-streamlit-app .

- docker run -p 8501:8501 my-streamlit-app

Extensively, if you want to deploy on the server with Traefik, change some names in docker-compose.yaml and run

- docker compose up -d


[Notice] Make sure your cloned repository contains Dockerfile, requirements.txt, scaler.pkl, car_price_prediction.model and model.pkl files.