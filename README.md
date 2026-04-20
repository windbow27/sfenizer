# Sfenizer

Convert a Shogi board into SFEN/CSA.

The dataset is pretty limited (I only have one Shogi board). Feel free to contact me for the dataset and train your own.

1. Create a virtual environment and install the requirements:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Download the model and place it in the root directory.

3. Start the backend:

   ```bash
   uvicorn api:app --reload
   ```

4. Start the frontend:

   ```bash
   cd web
   npm install
   npm run dev
   ```