# Problem

## 训推分离
trainer逻辑：先rollout再train
rollout时使用vllm进行推理，但不要setup_model
在train开始前，先setup_model，但train之后要卸载model，同时save_lora_adapter(str(lora_adapter_dir))


## 类加载
正确初始化各个类，trainer中RolloutCollector中未加载lora_adapter_path

