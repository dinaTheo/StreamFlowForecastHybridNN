###  Set up instructions

1. Connect to a HEX node and dump project
2. Add .csv files to Data folder
3. Build docker image in HEX, e.g., 
   ```hare build -t kt918/stream-frc-kym .```

4. Run docker container with the hybrid Scattering LSTM model, e.g.,
   ```
   hare run --rm
            --gpus device=4
            -v "$(pwd)":/app
            --name dina
            --user $(id -u):$(id -g)
            kt918/stream-frc-kym
            python3 Models/Kymatio/main_mul_scatt.py
   ```

