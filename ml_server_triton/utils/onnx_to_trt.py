import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_file_path, engine_file_path):
    with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH) as network, \
            trt.OnnxParser(network, TRT_LOGGER) as parser:
        # Создаем конфигурацию для построителя
        builder_config = builder.create_builder_config()
        builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

        # Включаем FP16 режим, если поддерживается
        if builder.platform_has_fast_fp16:
            builder_config.set_flag(trt.BuilderFlag.FP16)
        
        # Загрузка и парсинг ONNX модели
        with open(onnx_file_path, 'rb') as model:
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None

        # Построение CUDA движка
        engine = builder.build_engine(network, builder_config)

        # Сохранение CUDA движка
        with open(engine_file_path, 'wb') as f:
            f.write(engine.serialize())

# Пример использования
build_engine('resnet18.onnx', 'resnet18.trt')
