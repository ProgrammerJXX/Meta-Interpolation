import segyio
import numpy as np

def informations(file_path, num_shots=None, inlines=None, crosslines=None, offset=None, per_shot_trace_number=None):
    # 数据为shot-inlines-crosslines-samples 当shot为1时，我们默认为shot-inlines-1-samples
    with segyio.open(file_path, "r", ignore_geometry=True) as segyfile:
        
        # 获取道数(总地震道数量)
        num_traces = segyfile.tracecount
        traces = segyfile.trace.raw[:]
        
        # 道间距
        if offset == None:
            offset = 20
        
        trace_sequence_line = set()
        trace_sequence_file = set()
        
        field_record = set()
        energy_point = set()
        
        per_shot_trace_number = set()
        # inlines = set()
        # crosslines = set()
        sample_counts =set()
        sample_intervals = set()
        new_shot_trace_index = [0]
        offsets_x = set()
        offsets_y = set()
        for i in range(num_traces):
            trace_sequence_line.add(segyfile.header[i][segyio.TraceField.TRACE_SEQUENCE_LINE])
            trace_sequence_file.add(segyfile.header[i][segyio.TraceField.TRACE_SEQUENCE_FILE])
            field_record.add(segyfile.header[i][segyio.TraceField.FieldRecord])
            per_shot_trace_number.add(segyfile.header[i][segyio.TraceField.TraceNumber])
            energy_point.add(segyfile.header[i][segyio.TraceField.EnergySourcePoint])
            if i+1 < num_traces:
                offset_x = (segyfile.header[i][segyio.TraceField.GroupX] - segyfile.header[i+1][segyio.TraceField.GroupX])
                offsets_x.add(offset_x)
                offset_y = (segyfile.header[i][segyio.TraceField.GroupY] - segyfile.header[i+1][segyio.TraceField.GroupY])
                offsets_y.add(offset_y)
                if offset_x > offset*10:
                    new_shot_trace_index.append(i+1)
            # inlines.add(segyfile.header[i][segyio.TraceField.INLINE_3D])
            # crosslines.add(segyfile.header[i][segyio.TraceField.CROSSLINE_3D])
            sample_counts.add(segyfile.header[i][segyio.TraceField.TRACE_SAMPLE_COUNT])
            sample_intervals.add(segyfile.header[i][segyio.TraceField.TRACE_SAMPLE_INTERVAL])
        new_shot_trace_index.append(num_traces)
        new_shot_trace_index = [[new_shot_trace_index[i], new_shot_trace_index[i + 1]] for i in range(0, len(new_shot_trace_index) - 1)]
        real_per_shot_trace_number = [pair[1] - pair[0] for pair in new_shot_trace_index]
        print("new_shot_trace_index:", new_shot_trace_index)
        
        # 完整炮数
        num_shots_full = real_per_shot_trace_number.count(len(per_shot_trace_number))

        # GroupX差值集合
        offsets_x = sorted(list(offsets_x))

        # GroupY差值集合
        offsets_y = sorted(list(offsets_y))

        # 获取炮数&每炮道数
        if num_shots:
            num_shots = num_shots
        else:
            num_shots = max(len(field_record),len(energy_point))

        # 激发点编号
        field_record = len(field_record)
        energy_point = len(energy_point)

        # line(炮集)内的道序号&整个文件中的道序号(全局)
        trace_sequence_line = len(trace_sequence_line)
        trace_sequence_file = len(trace_sequence_file)

        # trace_number
        per_shot_trace_number = len(per_shot_trace_number)

        # inlines
        if inlines is None:
            inlines = min(per_shot_trace_number, trace_sequence_file)
        # else:
        #     ori_inlines = len(inlines)
            
        # crosslines
        if crosslines is None:
            if trace_sequence_line == trace_sequence_file:  # 如果整个文件中道序号和某一line中道序号一样，说明只有一条检波线 
                crosslines = 1
            else:
                crosslines = max(trace_sequence_file, per_shot_trace_number)/inlines
        # else:
        #     ori_crosslines = len(crosslines)

        # 获取每个地震道的采样点数(通过样本时间轴长度)
        num_samples = len(segyfile.samples)
        # num_samples = segyfile.header[0][segyio.TraceField.TRACE_SAMPLE_COUNT]

        # 获取采样间隔(单位：微秒)，并转换为毫秒
        sample_interval_us = segyfile.bin[segyio.BinField.Interval]
        sample_interval_ms = sample_interval_us / 1000.0
        # sample_interval_ms = list(sample_intervals)[0] / 1000.0)

    # 筛选出包含480个Trace的Shot
    valid_shots = []
    for shot, (start, end) in enumerate(new_shot_trace_index):
        if end - start == per_shot_trace_number:
            valid_shots.append((start, end))

    # 提取选定Shot的数据并组合为3D数组
    data_list = []
    for shot, (start, end) in enumerate(valid_shots):
        shot_data = traces[start:end]  # 形状为 [per_shot_trace_numbers, 采样点数]
        print(shot_data.shape)
        data_list.append(shot_data)
    
    # 合并为3D数组（198个Shot × 480个Trace × 采样点数）
    data_3d = np.stack(data_list, axis=0)


    print(f"道数(总地震道数量): {num_traces}")
    print(f"炮编号总数: {field_record}")
    print(f"激发点个数: {energy_point}")
    print(f"完整炮总数: {num_shots_full}")
    print(f"GroupX差值集合: {offsets_x}")
    print(f"GroupY差值集合: {offsets_y}")
    print(f"line(炮集)内的道序号: {trace_sequence_line}")
    print(f"整个文件中的道序号(全局): {trace_sequence_file}")
    print(f"每炮内共几道: {per_shot_trace_number}")
    print()
    # print(f"ori_inlines: {ori_inlines}")
    # print(f"ori_crosslines: {ori_crosslines}")
    print(f"每个地震道的采样点数: {num_samples}")
    print(f"采样间隔: {sample_interval_ms} 毫秒")
    
    print(f"炮数: {num_shots}")
    print(f"inlines: {inlines}")
    print(f"crosslines: {crosslines}")

    return new_shot_trace_index, data_3d


def find_zero_ranges(arr, zero_threshold=50):
    # 确保输入是三维数组
    if arr.ndim != 3:
        raise ValueError("Input array must be 3-dimensional.")
    
    # 获取数组的形状
    batch, h, w = arr.shape
    
    # 存储结果
    zero_ranges = []

    # 遍历每个batch
    for b in range(batch):
        # 获取当前batch的二维数组
        current_slice = arr[b]
        
        # 检测每一行
        for row in range(h):
            # 找到当前行的零值索引
            zero_indices = np.where(current_slice[row] == 0)[0]
            
            # 检查零值的连续性
            if len(zero_indices) > 0:
                # 计算连续零值的起始和结束索引
                start_idx = zero_indices[0]
                count = 1
                
                for i in range(1, len(zero_indices)):
                    if zero_indices[i] == zero_indices[i - 1] + 1:
                        count += 1
                    else:
                        if count > zero_threshold:  # 大于zero_threshold个为零
                            zero_ranges.append((b, row, start_idx, start_idx + count - 1))
                        start_idx = zero_indices[i]
                        count = 1
                
                # 检查最后一段
                if count > zero_threshold:
                    zero_ranges.append((b, row, start_idx, start_idx + count - 1))

    return zero_ranges


if __name__ == '__main__':
    
    # # SEGC3
    # print('SEGC3')
    # file_path = 'G:\seismic data\Hint_seismic\data\SEG C3\SEG_45Shot_shots1_9.sgy'
    # new_shot_trace_index, data_3d = informations(file_path, num_shots=9, inlines=201, crosslines=201, offset=20, per_shot_trace_number=None)
    # print('\n')

    # # MAVO
    # print('MAVO')
    # file_path = 'G:\seismic data\Hint_seismic\data\MAVG\Mobil_Avo_Viking_Graben_Line_12.segy'
    # new_shot_trace_index, data_3d = informations(file_path)
    # print('\n')
    
    # # MODEL94
    # print('MODEL94')
    # file_path = 'G:\seismic data\Hint_seismic\data\MODEL94\Model94_shots.segy'
    # new_shot_trace_index, data_3d = informations(file_path, num_shots=278, per_shot_trace_number=480)
    # print('\n')

    # F3_Netherlands
    print('F3_Netherlands')
    file_path = 'H:\seismic data\Hint_seismic\data\F3_Netherlands\Seismic_data.sgy'
    new_shot_trace_index, data_3d = informations(file_path, num_shots=1, inlines=601, crosslines=951, offset=None, per_shot_trace_number=None)
    print('\n')

    # 查找零值范围
    zero_ranges = find_zero_ranges(data_3d, zero_threshold=20)

    max_first_zero_Index = []
    min_last_zero_Index = []

    # 输出结果
    for b, row, start, end in zero_ranges:
        print(f"Batch: {b}, Row: {row}, Start Index: {start}, End Index: {end}")
        if start == 0:
            max_first_zero_Index.append(end)
        if end == 1999:
            min_last_zero_Index.append(start)

    print(f"max_first_zero_Index: {max(max_first_zero_Index)}, min_last_zero_Index: {min(min_last_zero_Index)}")