from dp_cgans import DP_CGAN
import pandas as pd

# Load the training data
data = pd.read_csv("data/adult/train.csv")

# Initialize DP-CGAN with your custom hyperparameters
model = DP_CGAN(
    epochs=100,
    batch_size=100,
    generator_dim=(128, 128, 128),
    discriminator_dim=(128, 128, 128),
    generator_lr=2e-4,
    discriminator_lr=2e-4,
    private=True,  # enable differential privacy
)

# Train the model
print("🧠 Training model... this might take a few mins ⏳")
model.fit(data)

# Generate 100 synthetic samples
print("✨ Generating synthetic data...")
synthetic_data = model.sample(100)

# Save the synthetic data
synthetic_data.to_csv("synthetic_data.csv", index=False)
print("✅ Done! Synthetic data saved as synthetic_data.csv")
