# app.py – Upgraded Streamlit UI
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
import time
from sklearn.preprocessing import MinMaxScaler
from sensor_reader import read_hardware_data as generate_data

class VAE(nn.Module):
    def __init__(self, input_dim=3, latent_dim=2):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc21 = nn.Linear(16, latent_dim)
        self.fc22 = nn.Linear(16, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 16)
        self.fc4 = nn.Linear(16, input_dim)

    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        return self.fc4(h3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

scaler = MinMaxScaler()
training_data = np.array([[random.uniform(90, 98), random.randint(60, 90), random.randint(12, 20)] for _ in range(500)])
training_data = scaler.fit_transform(training_data)

vae = VAE()
optimizer = optim.Adam(vae.parameters(), lr=0.001)

def train_vae(model, data, epochs=10):
    for epoch in range(epochs):
        for x in data:
            x = torch.tensor(x, dtype=torch.float32)
            optimizer.zero_grad()
            recon, mu, logvar = model(x)
            loss = loss_function(recon, x, mu, logvar)
            loss.backward()
            optimizer.step()

train_vae(vae, training_data)

st.set_page_config(page_title="Smart Band Dashboard", layout="wide")
st.title("🩺 Smart Band - ILD Patient Monitoring")
st.markdown("AI-powered real-time vitals tracking with anomaly detection")

st.sidebar.title("📋 Patient Details")
st.sidebar.markdown("""
**Name:** John Doe  
**Age:** 62  
**Gender:** Male  
**Patient ID:** ILD-2032  
**Condition:** Interstitial Lung Disease (ILD)  
**Sensor Used:** MAX30102  
**Monitoring Device:** Smart Band v1.2  
**AI Model:** VAE (Anomaly Detection)  
**Session:** Live Monitoring  
""")

spo2_vals, timestamps = [], []
placeholder = st.empty()

for _ in range(30):
    try:
        data = generate_data()
        spo2 = data['spo2']
        hr = data['heart_rate']
        rr = data['resp_rate']

        input_vals = scaler.transform([[spo2, hr, rr]])[0]
        input_tensor = torch.tensor(input_vals, dtype=torch.float32)
        recon, mu, logvar = vae(input_tensor)
        loss = loss_function(recon, input_tensor, mu, logvar).item()
        anomaly = loss > 5.0

        spo2_vals.append(spo2)
        timestamps.append(time.strftime("%H:%M:%S"))

        with placeholder.container():
            col1, col2, col3 = st.columns(3)
            col1.metric("SpO₂ (%)", f"{spo2:.1f}", delta=None)
            col2.metric("Heart Rate (bpm)", f"{hr:.1f}")
            col3.metric("Respiratory Rate", f"{rr} bpm")

            if anomaly:
                st.error("⚠️ Anomaly Detected - Possible Oxygen Desaturation!", icon="🚨")
            else:
                st.success("✅ Vitals Normal")

            st.line_chart(pd.DataFrame({"SpO₂": spo2_vals}, index=timestamps))

        time.sleep(1)

    except Exception as e:
        st.warning(f"⚠️ Error reading sensor or fallback triggered: {e}")
        break

st.markdown("---")
st.markdown("Made with ❤️ by Mohamed Sarfraz | SRM CSE-AIML | MAX30102 + VAE-GAN Hybrid System")