  if(!gnn_context_->setInputShape(desc_name, desc_dims)) {
            auto engine_dims = gnn_engine_->getTensorShape(desc_name);
            std::cerr << "Descriptors维度绑定失败! 引擎期望: ";
            for (int i=0; i<engine_dims.nbDims; ++i) 
                std::cerr << (engine_dims.d[i]==-1 ? "?" : std::to_string(engine_dims.d[i])) << " ";
            std::cerr << "\n实际设置: ";
            for (int i=0; i<desc_dims.nbDims; ++i) 
                std::cerr << desc_dims.d[i] << " ";
            continue; // 跳过当前batch
        }

        if (!gnn_context_->setInputShape(points_name, points_dims)) {
            auto engine_dims = gnn_engine_->getTensorShape(points_name);
            std::cerr << "Points维度绑定失败! 引擎期望: ";
            for (int i=0; i<engine_dims.nbDims; ++i) 
                std::cerr << (engine_dims.d[i]==-1 ? "?" : std::to_string(engine_dims.d[i])) << " ";
            std::cerr << "\n实际设置: ";
            for (int i=0; i<points_dims.nbDims; ++i) 
                std::cerr << points_dims.d[i] << " ";
            continue;
        }