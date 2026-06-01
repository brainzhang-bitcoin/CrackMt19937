# -*- coding: utf-8 -*-

import pyopencl as cl
import numpy as np

def test_opencl():
    try:
        print("=== 正在探测 OpenCL 平台 ===")
        platforms = cl.get_platforms()
        if not platforms:
            print("错误: 未找到任何 OpenCL Platform (驱动未加载)")
            return

        for p in platforms:
            print(f"平台名称: {p.name}")
            devices = p.get_devices()
            for d in devices:
                print(f"  - 设备名称: {d.name}")
                print(f"  - 设备类型: {cl.device_type.to_string(d.type)}")
                print(f"  - 驱动版本: {d.driver_version}")

        # 尝试创建一个上下文 (Context) - 这是你卡住的地方
        print("\n=== 尝试创建 Context (初始化硬件) ===")
        ctx = cl.Context(devices=[devices[0]])
        print("成功: Context 已创建!")

        # 尝试编译一个简单的 Kernel
        print("=== 尝试编译简单 Kernel ===")
        queue = cl.CommandQueue(ctx)
        prg = cl.Program(ctx, """
            __kernel void add(__global const float *a, __global float *res) {
              int i = get_global_id(0);
              res[i] = a[i] + 100.0f;
            }
        """).build()
        print("成功: Kernel 编译完成!")

    except Exception as e:
        print(f"\n[!!!] 运行出错: {e}")

if __name__ == "__main__":
    test_opencl()
