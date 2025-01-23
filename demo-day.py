import exla

orig_model = "meta-llama/Llama-3.1-8B-Instruct" 

optimized_model = exla.optimize(orig_model, target_hardware="jetson-orin-nano")

orig_model_server = exla.server(orig_model, port=8080)
optimized_model_server = exla.server(optimized_model, port=8080)


orig_model_server.start()
# orig_model_server.stop()


# optimized_model_server.start()
# optimized_model_server.stop()
