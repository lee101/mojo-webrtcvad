"""Fixed-point WebRTC voice activity detector with a C ABI.

The detector state and scratch memory live in a caller-owned Int32 buffer.
Audio crosses the ABI as an Int16 buffer address.
"""

from std.sys.info import simd_width_of

comptime I16Ptr = UnsafePointer[Int16, AnyOrigin[mut=True]]
comptime I32Ptr = UnsafePointer[Int32, AnyOrigin[mut=True]]
comptime U8Ptr = UnsafePointer[UInt8, AnyOrigin[mut=True]]

comptime FRAME_COUNTER = 0
comptime OVER_HANG = 1
comptime NUM_SPEECH = 2
comptime DOWNSAMPLE = 3
comptime NOISE_MEANS = 7
comptime SPEECH_MEANS = 19
comptime NOISE_STDS = 31
comptime SPEECH_STDS = 43
comptime AGE = 55
comptime LOW_VALUES = 151
comptime MEAN_VALUE = 247
comptime UPPER_STATE = 253
comptime LOWER_STATE = 258
comptime HP_STATE = 263
comptime OVER1 = 267
comptime OVER2 = 270
comptime INDIVIDUAL = 273
comptime TOTAL = 276
comptime RESAMPLE48 = 279

comptime DATA8 = 400
comptime HP120 = 640
comptime LP120 = 760
comptime HP60 = 880
comptime LP60 = 940
comptime FEATURES = 1000
comptime DELTA_N = 1006
comptime DELTA_S = 1018
comptime NGPR = 1030
comptime SGPR = 1042
comptime PROB_N = 1054
comptime PROB_S = 1056
comptime TEMP32 = 1120


def s16(x: Int) -> Int:
    return Int(Int16(x))


def w32(x: Int) -> Int:
    return Int(Int32(x))


def sat16(x: Int) -> Int:
    if x > 32767:
        return 32767
    if x < -32768:
        return -32768
    return x


def norm_w32(a: Int) -> Int:
    if a == 0:
        return 0
    var x = a
    if x < 0:
        x = w32(~x)
    var shifts = 0
    while x < 0x40000000:
        x <<= 1
        shifts += 1
    return shifts


def norm_u32(a: Int) -> Int:
    if a == 0:
        return 0
    var x = a
    var shifts = 0
    while x < 0x80000000:
        x <<= 1
        shifts += 1
    return shifts


def size_in_bits(a: Int) -> Int:
    if a == 0:
        return 0
    return 32 - norm_u32(a)


def noise_weight(i: Int) -> Int:
    if i == 0: return 34
    if i == 1: return 62
    if i == 2: return 72
    if i == 3: return 66
    if i == 4: return 53
    if i == 5: return 25
    if i == 6: return 94
    if i == 7: return 66
    if i == 8: return 56
    if i == 9: return 62
    if i == 10: return 75
    return 103


def speech_weight(i: Int) -> Int:
    if i == 0: return 48
    if i == 1: return 82
    if i == 2: return 45
    if i == 3: return 87
    if i == 4: return 50
    if i == 5: return 47
    if i == 6: return 80
    if i == 7: return 46
    if i == 8: return 83
    if i == 9: return 41
    if i == 10: return 78
    return 81


def noise_mean_initial(i: Int) -> Int:
    if i == 0: return 6738
    if i == 1: return 4892
    if i == 2: return 7065
    if i == 3: return 6715
    if i == 4: return 6771
    if i == 5: return 3369
    if i == 6: return 7646
    if i == 7: return 3863
    if i == 8: return 7820
    if i == 9: return 7266
    if i == 10: return 5020
    return 4362


def speech_mean_initial(i: Int) -> Int:
    if i == 0: return 8306
    if i == 1: return 10085
    if i == 2: return 10078
    if i == 3: return 11823
    if i == 4: return 11843
    if i == 5: return 6309
    if i == 6: return 9473
    if i == 7: return 9571
    if i == 8: return 10879
    if i == 9: return 7581
    if i == 10: return 8180
    return 7483


def noise_std_initial(i: Int) -> Int:
    if i == 0: return 378
    if i == 1: return 1064
    if i == 2: return 493
    if i == 3: return 582
    if i == 4: return 688
    if i == 5: return 593
    if i == 6: return 474
    if i == 7: return 697
    if i == 8: return 475
    if i == 9: return 688
    if i == 10: return 421
    return 455


def speech_std_initial(i: Int) -> Int:
    if i == 0: return 555
    if i == 1: return 505
    if i == 2: return 567
    if i == 3: return 524
    if i == 4: return 585
    if i == 5: return 1231
    if i == 6: return 509
    if i == 7: return 828
    if i == 8: return 492
    if i == 9: return 1540
    if i == 10: return 1079
    return 850


def spectrum_weight(channel: Int) -> Int:
    return 6 + 2 * channel


def minimum_difference(channel: Int) -> Int:
    if channel < 2:
        return 544
    return 576


def maximum_speech(channel: Int) -> Int:
    if channel < 2:
        return 11392
    return 11520


def maximum_noise(channel: Int) -> Int:
    return 9216 - 128 * channel


def allpass_coef(branch: Int) -> Int:
    if branch == 0:
        return 20972
    return 5571


def offset_value(channel: Int) -> Int:
    if channel < 2:
        return 368
    if channel == 2:
        return 272
    return 176


def split_filter(
    s: I32Ptr, input_offset: Int, length: Int, upper_offset: Int,
    lower_offset: Int, hp_offset: Int, lp_offset: Int
):
    var half = length >> 1
    var state32 = Int(s[upper_offset]) << 16
    for i in range(half):
        var sample = Int(s[input_offset + 2 * i])
        var tmp32 = state32 + allpass_coef(0) * sample
        var tmp16 = s16(tmp32 >> 16)
        s[hp_offset + i] = Int32(tmp16)
        state32 = (sample << 14) - allpass_coef(0) * tmp16
        state32 *= 2
    s[upper_offset] = Int32(s16(state32 >> 16))

    state32 = Int(s[lower_offset]) << 16
    for i in range(half):
        var sample = Int(s[input_offset + 2 * i + 1])
        var tmp32 = state32 + allpass_coef(1) * sample
        var tmp16 = s16(tmp32 >> 16)
        s[lp_offset + i] = Int32(tmp16)
        state32 = (sample << 14) - allpass_coef(1) * tmp16
        state32 *= 2
    s[lower_offset] = Int32(s16(state32 >> 16))

    comptime W = simd_width_of[DType.int32]()
    var i = 0
    while i + W <= half:
        var upper = s.load[width=W](hp_offset + i)
        var lower = s.load[width=W](lp_offset + i)
        var high = (upper - lower) << 16
        var low = (lower + upper) << 16
        s.store(hp_offset + i, high >> 16)
        s.store(lp_offset + i, low >> 16)
        i += W
    while i < half:
        var upper = Int(s[hp_offset + i])
        var lower = Int(s[lp_offset + i])
        s[hp_offset + i] = Int32(s16(upper - lower))
        s[lp_offset + i] = Int32(s16(lower + upper))
        i += 1


def high_pass(s: I32Ptr, input_offset: Int, length: Int, output_offset: Int):
    for i in range(length):
        var sample = Int(s[input_offset + i])
        var tmp32 = 6631 * sample
        tmp32 += -13262 * Int(s[HP_STATE])
        tmp32 += 6631 * Int(s[HP_STATE + 1])
        s[HP_STATE + 1] = s[HP_STATE]
        s[HP_STATE] = Int32(sample)
        tmp32 -= -7756 * Int(s[HP_STATE + 2])
        tmp32 -= 5620 * Int(s[HP_STATE + 3])
        s[HP_STATE + 3] = s[HP_STATE + 2]
        var filtered = s16(tmp32 >> 14)
        s[HP_STATE + 2] = Int32(filtered)
        s[output_offset + i] = Int32(filtered)


def energy(s: I32Ptr, input_offset: Int, length: Int, scale_offset: Int) -> Int:
    var maximum = -1
    comptime W = simd_width_of[DType.int32]()
    var i = 0
    while i + W <= length:
        var values = s.load[width=W](input_offset + i)
        var absolute = max(values, -values)
        for lane in range(W):
            if Int(absolute[lane]) > maximum:
                maximum = Int(absolute[lane])
        i += W
    while i < length:
        var value = Int(s[input_offset + i])
        var absolute = value if value > 0 else -value
        if absolute > maximum:
            maximum = absolute
        i += 1
    if maximum == 0:
        s[scale_offset] = 0
        return 0
    var nbits = size_in_bits(length)
    var normal = norm_w32(maximum * maximum)
    var scaling = 0 if normal > nbits else nbits - normal
    var total_energy = 0
    comptime SUM_W = simd_width_of[DType.int64]()
    var sums = SIMD[DType.int64, SUM_W](0)
    i = 0
    while i + SUM_W <= length:
        var energy_values = s.load[width=SUM_W](input_offset + i).cast[DType.int64]()
        sums += (energy_values * energy_values) >> SIMD[DType.int64, SUM_W](scaling)
        i += SUM_W
    total_energy = Int(sums.reduce_add())
    while i < length:
        var value = Int(s[input_offset + i])
        total_energy += (value * value) >> scaling
        i += 1
    s[scale_offset] = Int32(scaling)
    return total_energy


def log_of_energy(
    s: I32Ptr, input_offset: Int, length: Int, offset: Int,
    total_energy_offset: Int, feature_offset: Int
):
    var raw_energy = energy(s, input_offset, length, TEMP32)
    var total_rshifts = Int(s[TEMP32])
    if raw_energy == 0:
        s[feature_offset] = Int32(offset)
        return
    var normalizing = 17 - norm_u32(raw_energy)
    total_rshifts += normalizing
    var normalized = raw_energy
    if normalizing < 0:
        normalized <<= -normalizing
    else:
        normalized >>= normalizing
    var log2_energy = 14336 + ((normalized & 0x3FFF) >> 4)
    var log_energy = ((24660 * log2_energy) >> 19) + ((total_rshifts * 24660) >> 9)
    if log_energy < 0:
        log_energy = 0
    log_energy = s16(log_energy + offset)
    s[feature_offset] = Int32(log_energy)
    var total_energy = Int(s[total_energy_offset])
    if total_energy <= 10:
        if total_rshifts >= 0:
            total_energy += 11
        else:
            total_energy += normalized >> -total_rshifts
        s[total_energy_offset] = Int32(s16(total_energy))


def calculate_features(s: I32Ptr, length: Int) -> Int:
    s[TEMP32 + 1] = 0
    var half = length >> 1
    split_filter(s, DATA8, length, UPPER_STATE, LOWER_STATE, HP120, LP120)
    split_filter(s, HP120, half, UPPER_STATE + 1, LOWER_STATE + 1, HP60, LP60)
    var quarter = half >> 1
    log_of_energy(s, HP60, quarter, offset_value(5), TEMP32 + 1, FEATURES + 5)
    log_of_energy(s, LP60, quarter, offset_value(4), TEMP32 + 1, FEATURES + 4)
    split_filter(s, LP120, half, UPPER_STATE + 2, LOWER_STATE + 2, HP60, LP60)
    log_of_energy(s, HP60, quarter, offset_value(3), TEMP32 + 1, FEATURES + 3)
    split_filter(s, LP60, quarter, UPPER_STATE + 3, LOWER_STATE + 3, HP120, LP120)
    var eighth = quarter >> 1
    log_of_energy(s, HP120, eighth, offset_value(2), TEMP32 + 1, FEATURES + 2)
    split_filter(s, LP120, eighth, UPPER_STATE + 4, LOWER_STATE + 4, HP60, LP60)
    var sixteenth = eighth >> 1
    log_of_energy(s, HP60, sixteenth, offset_value(1), TEMP32 + 1, FEATURES + 1)
    high_pass(s, LP60, sixteenth, HP120)
    log_of_energy(s, HP120, sixteenth, offset_value(0), TEMP32 + 1, FEATURES)
    return Int(s[TEMP32 + 1])


def gaussian_probability(input_value: Int, mean: Int, std: Int, delta_offset: Int, s: I32Ptr) -> Int:
    var inv_std = s16((131072 + (std >> 1)) / std)
    var tmp16 = inv_std >> 2
    var inv_std2 = s16((tmp16 * tmp16) >> 2)
    tmp16 = s16((input_value << 3) - mean)
    var delta = s16((inv_std2 * tmp16) >> 10)
    s[delta_offset] = Int32(delta)
    var exponent = (delta * tmp16) >> 9
    var exp_value = 0
    if exponent < 22005:
        tmp16 = s16((5909 * exponent) >> 12)
        tmp16 = s16(-tmp16)
        exp_value = s16(0x0400 | (tmp16 & 0x03FF))
        tmp16 = s16(tmp16 ^ 0xFFFF)
        tmp16 >>= 10
        tmp16 += 1
        exp_value >>= tmp16
    return inv_std * exp_value


def find_minimum(s: I32Ptr, feature: Int, channel: Int) -> Int:
    var base = channel << 4
    var i = 0
    while i < 16:
        var age_value = Int(s[AGE + base + i])
        if age_value != 100:
            s[AGE + base + i] = Int32(s16(age_value + 1))
        else:
            var j = i
            while j < 15:
                s[LOW_VALUES + base + j] = s[LOW_VALUES + base + j + 1]
                s[AGE + base + j] = s[AGE + base + j + 1]
                j += 1
            s[AGE + base + 15] = 101
            s[LOW_VALUES + base + 15] = 10000
        i += 1

    var position = -1
    for k in range(16):
        if position < 0 and feature < Int(s[LOW_VALUES + base + k]):
            position = k
    if position >= 0:
        i = 15
        while i > position:
            s[LOW_VALUES + base + i] = s[LOW_VALUES + base + i - 1]
            s[AGE + base + i] = s[AGE + base + i - 1]
            i -= 1
        s[LOW_VALUES + base + position] = Int32(feature)
        s[AGE + base + position] = 1

    var current_median = 1600
    var frame_count = Int(s[FRAME_COUNTER])
    if frame_count > 2:
        current_median = Int(s[LOW_VALUES + base + 2])
    elif frame_count > 0:
        current_median = Int(s[LOW_VALUES + base])
    var alpha = 0
    var old_mean = Int(s[MEAN_VALUE + channel])
    if frame_count > 0:
        alpha = 6553 if current_median < old_mean else 32439
    var tmp32 = (alpha + 1) * old_mean
    tmp32 += (32767 - alpha) * current_median
    tmp32 += 16384
    var result = s16(tmp32 >> 15)
    s[MEAN_VALUE + channel] = Int32(result)
    return result


def weighted_average(s: I32Ptr, base: Int, channel: Int, offset: Int, speech: Bool) -> Int:
    var result = 0
    for k in range(2):
        var index = channel + k * 6
        var value = s16(Int(s[base + index]) + offset)
        s[base + index] = Int32(value)
        var weight = speech_weight(index) if speech else noise_weight(index)
        result += value * weight
    return result


def gmm_probability(s: I32Ptr, total_power: Int, frame_length: Int) -> Int:
    var frame_index = 2
    if frame_length == 80:
        frame_index = 0
    elif frame_length == 160:
        frame_index = 1
    var overhead1 = Int(s[OVER1 + frame_index])
    var overhead2 = Int(s[OVER2 + frame_index])
    var individual_test = Int(s[INDIVIDUAL + frame_index])
    var total_test = Int(s[TOTAL + frame_index])
    var vadflag = 0

    if total_power > 10:
        for i in range(12):
            s[NGPR + i] = 0
            s[SGPR + i] = 0
        var likelihood_sum = 0
        for channel in range(6):
            var h0_test = 0
            var h1_test = 0
            for k in range(2):
                var gaussian = channel + k * 6
                var p0 = gaussian_probability(
                    Int(s[FEATURES + channel]), Int(s[NOISE_MEANS + gaussian]),
                    Int(s[NOISE_STDS + gaussian]), DELTA_N + gaussian, s
                )
                var noise_probability = noise_weight(gaussian) * p0
                s[PROB_N + k] = Int32(noise_probability)
                h0_test += noise_probability
                var p1 = gaussian_probability(
                    Int(s[FEATURES + channel]), Int(s[SPEECH_MEANS + gaussian]),
                    Int(s[SPEECH_STDS + gaussian]), DELTA_S + gaussian, s
                )
                var speech_probability = speech_weight(gaussian) * p1
                s[PROB_S + k] = Int32(speech_probability)
                h1_test += speech_probability

            var shifts_h0 = norm_w32(h0_test)
            var shifts_h1 = norm_w32(h1_test)
            if h0_test == 0:
                shifts_h0 = 31
            if h1_test == 0:
                shifts_h1 = 31
            var ratio = s16(shifts_h0 - shifts_h1)
            likelihood_sum += ratio * spectrum_weight(channel)
            if ratio * 4 > individual_test:
                vadflag = 1

            var h0 = s16(h0_test >> 12)
            if h0 > 0:
                var numerator = w32((Int(s[PROB_N]) & 0xFFFFF000) << 2)
                var probability = s16(numerator / h0)
                s[NGPR + channel] = Int32(probability)
                s[NGPR + channel + 6] = Int32(s16(16384 - probability))
            else:
                s[NGPR + channel] = 16384
            var h1 = s16(h1_test >> 12)
            if h1 > 0:
                var numerator = w32((Int(s[PROB_S]) & 0xFFFFF000) << 2)
                var probability = s16(numerator / h1)
                s[SGPR + channel] = Int32(probability)
                s[SGPR + channel + 6] = Int32(s16(16384 - probability))

        if likelihood_sum >= total_test:
            vadflag = 1

        var max_speech = 12800
        for channel in range(6):
            var feature_minimum = find_minimum(s, Int(s[FEATURES + channel]), channel)
            var noise_global = weighted_average(s, NOISE_MEANS, channel, 0, False)
            var noise_global_q8 = s16(noise_global >> 6)
            for k in range(2):
                var gaussian = channel + k * 6
                var nmk = Int(s[NOISE_MEANS + gaussian])
                var smk = Int(s[SPEECH_MEANS + gaussian])
                var nsk = Int(s[NOISE_STDS + gaussian])
                var ssk = Int(s[SPEECH_STDS + gaussian])
                var nmk2 = nmk
                if vadflag == 0:
                    var delt = s16((Int(s[NGPR + gaussian]) * Int(s[DELTA_N + gaussian])) >> 11)
                    nmk2 = s16(nmk + ((delt * 655) >> 22))
                var ndelt = s16((feature_minimum << 4) - noise_global_q8)
                var nmk3 = s16(nmk2 + ((ndelt * 154) >> 9))
                var lower = (k + 5) << 7
                if nmk3 < lower:
                    nmk3 = lower
                var upper = (72 + k - channel) << 7
                if nmk3 > upper:
                    nmk3 = upper
                s[NOISE_MEANS + gaussian] = Int32(nmk3)

                if vadflag != 0:
                    var delt = s16((Int(s[SGPR + gaussian]) * Int(s[DELTA_S + gaussian])) >> 11)
                    var tmp16 = s16((delt * 6554) >> 21)
                    var smk2 = s16(smk + ((tmp16 + 1) >> 1))
                    var maxmu = max_speech + 640
                    var minmu = 640 if k == 0 else 768
                    if smk2 < minmu:
                        smk2 = minmu
                    if smk2 > maxmu:
                        smk2 = maxmu
                    s[SPEECH_MEANS + gaussian] = Int32(smk2)
                    tmp16 = (smk + 4) >> 3
                    tmp16 = s16(Int(s[FEATURES + channel]) - tmp16)
                    var tmp1 = (Int(s[DELTA_S + gaussian]) * tmp16) >> 3
                    var tmp2 = tmp1 - 4096
                    tmp16 = Int(s[SGPR + gaussian]) >> 2
                    tmp1 = tmp16 * tmp2
                    tmp2 = tmp1 >> 4
                    if tmp2 > 0:
                        tmp16 = s16(tmp2 / (ssk * 10))
                    else:
                        tmp16 = s16(-(s16((-tmp2) / (ssk * 10))))
                    tmp16 = s16(tmp16 + 128)
                    ssk = s16(ssk + (tmp16 >> 8))
                    if ssk < 384:
                        ssk = 384
                    s[SPEECH_STDS + gaussian] = Int32(ssk)
                else:
                    var tmp16 = s16(Int(s[FEATURES + channel]) - (nmk >> 3))
                    var tmp1 = (Int(s[DELTA_N + gaussian]) * tmp16) >> 3
                    tmp1 -= 4096
                    tmp16 = (Int(s[NGPR + gaussian]) + 2) >> 2
                    var tmp2 = w32(tmp16 * tmp1)
                    tmp1 = tmp2 >> 14
                    if tmp1 > 0:
                        tmp16 = s16(tmp1 / nsk)
                    else:
                        tmp16 = s16(-(s16((-tmp1) / nsk)))
                    tmp16 = s16(tmp16 + 32)
                    nsk = s16(nsk + (tmp16 >> 6))
                    if nsk < 384:
                        nsk = 384
                    s[NOISE_STDS + gaussian] = Int32(nsk)

            noise_global = weighted_average(s, NOISE_MEANS, channel, 0, False)
            var speech_global = weighted_average(s, SPEECH_MEANS, channel, 0, True)
            var difference = s16(s16(speech_global >> 9) - s16(noise_global >> 9))
            var minimum = minimum_difference(channel)
            if difference < minimum:
                var tmp16 = s16(minimum - difference)
                var speech_offset = s16((13 * tmp16) >> 2)
                var noise_offset = s16((3 * tmp16) >> 2)
                speech_global = weighted_average(s, SPEECH_MEANS, channel, speech_offset, True)
                noise_global = weighted_average(s, NOISE_MEANS, channel, -noise_offset, False)
            max_speech = maximum_speech(channel)
            var correction = s16(speech_global >> 7)
            if correction > max_speech:
                correction = s16(correction - max_speech)
                for k in range(2):
                    var idx = SPEECH_MEANS + channel + k * 6
                    s[idx] = Int32(s16(Int(s[idx]) - correction))
            correction = s16(noise_global >> 7)
            var max_noise = maximum_noise(channel)
            if correction > max_noise:
                correction = s16(correction - max_noise)
                for k in range(2):
                    var idx = NOISE_MEANS + channel + k * 6
                    s[idx] = Int32(s16(Int(s[idx]) - correction))
        s[FRAME_COUNTER] = Int32(Int(s[FRAME_COUNTER]) + 1)

    if vadflag == 0:
        if Int(s[OVER_HANG]) > 0:
            vadflag = 2 + Int(s[OVER_HANG])
            s[OVER_HANG] = Int32(Int(s[OVER_HANG]) - 1)
        s[NUM_SPEECH] = 0
    else:
        var speech_frames = Int(s[NUM_SPEECH]) + 1
        if speech_frames > 6:
            speech_frames = 6
            s[OVER_HANG] = Int32(overhead2)
        else:
            s[OVER_HANG] = Int32(overhead1)
        s[NUM_SPEECH] = Int32(speech_frames)
    return vadflag


def downsample_by_2(s: I32Ptr, input_offset: Int, output_offset: Int, state_offset: Int, length: Int):
    var state0 = Int(s[state_offset])
    var state1 = Int(s[state_offset + 1])
    var half = length >> 1
    for i in range(half):
        var sample0 = Int(s[input_offset + 2 * i])
        var tmp1 = s16((state0 >> 1) + ((5243 * sample0) >> 14))
        state0 = sample0 - ((5243 * tmp1) >> 12)
        var sample1 = Int(s[input_offset + 2 * i + 1])
        var tmp2 = s16((state1 >> 1) + ((1392 * sample1) >> 14))
        state1 = sample1 - ((1392 * tmp2) >> 12)
        s[output_offset + i] = Int32(s16(tmp1 + tmp2))
    s[state_offset] = Int32(state0)
    s[state_offset + 1] = Int32(state1)


def resample_allpass(branch: Int, stage: Int) -> Int:
    if branch == 0:
        if stage == 0: return 821
        if stage == 1: return 6110
        return 12382
    if stage == 0: return 3050
    if stage == 1: return 9368
    return 15063


def trunc_shift14(value: Int) -> Int:
    var result = w32(value) >> 14
    if result < 0:
        result += 1
    return result


def down_short_to_int(
    s: I32Ptr, input_offset: Int, output_offset: Int, state_offset: Int, length: Int
):
    var half = length >> 1
    var state0 = Int(s[state_offset])
    var state1 = Int(s[state_offset + 1])
    var state2 = Int(s[state_offset + 2])
    var state3 = Int(s[state_offset + 3])
    for i in range(half):
        var tmp0 = w32((Int(s[input_offset + 2 * i]) << 15) + 16384)
        var diff = w32(tmp0 - state1)
        diff = w32(diff + 8192) >> 14
        var tmp1 = w32(state0 + w32(diff * resample_allpass(1, 0)))
        state0 = tmp0
        diff = trunc_shift14(w32(tmp1 - state2))
        tmp0 = w32(state1 + w32(diff * resample_allpass(1, 1)))
        state1 = tmp1
        diff = trunc_shift14(w32(tmp0 - state3))
        state3 = w32(state2 + w32(diff * resample_allpass(1, 2)))
        state2 = tmp0
        s[output_offset + i] = Int32(state3 >> 1)
    s[state_offset] = Int32(state0)
    s[state_offset + 1] = Int32(state1)
    s[state_offset + 2] = Int32(state2)
    s[state_offset + 3] = Int32(state3)

    var state4 = Int(s[state_offset + 4])
    var state5 = Int(s[state_offset + 5])
    var state6 = Int(s[state_offset + 6])
    var state7 = Int(s[state_offset + 7])
    for i in range(half):
        var tmp0 = w32((Int(s[input_offset + 2 * i + 1]) << 15) + 16384)
        var diff = w32(tmp0 - state5)
        diff = w32(diff + 8192) >> 14
        var tmp1 = w32(state4 + w32(diff * resample_allpass(0, 0)))
        state4 = tmp0
        diff = trunc_shift14(w32(tmp1 - state6))
        tmp0 = w32(state5 + w32(diff * resample_allpass(0, 1)))
        state5 = tmp1
        diff = trunc_shift14(w32(tmp0 - state7))
        state7 = w32(state6 + w32(diff * resample_allpass(0, 2)))
        state6 = tmp0
        s[output_offset + i] = Int32(
            w32(Int(s[output_offset + i]) + (state7 >> 1))
        )
    s[state_offset + 4] = Int32(state4)
    s[state_offset + 5] = Int32(state5)
    s[state_offset + 6] = Int32(state6)
    s[state_offset + 7] = Int32(state7)


def lp_int_to_int(
    s: I32Ptr, input_offset: Int, output_offset: Int, state_offset: Int, length: Int
):
    var half = length >> 1
    var tmp0 = Int(s[state_offset + 12])
    var state0 = Int(s[state_offset])
    var state1 = Int(s[state_offset + 1])
    var state2 = Int(s[state_offset + 2])
    var state3 = Int(s[state_offset + 3])
    for i in range(half):
        var diff = w32(tmp0 - state1)
        diff = w32(diff + 8192) >> 14
        var tmp1 = w32(state0 + w32(diff * resample_allpass(1, 0)))
        state0 = tmp0
        diff = trunc_shift14(w32(tmp1 - state2))
        tmp0 = w32(state1 + w32(diff * resample_allpass(1, 1)))
        state1 = tmp1
        diff = trunc_shift14(w32(tmp0 - state3))
        state3 = w32(state2 + w32(diff * resample_allpass(1, 2)))
        state2 = tmp0
        s[output_offset + 2 * i] = Int32(state3 >> 1)
        tmp0 = Int(s[input_offset + 2 * i + 1])
    s[state_offset] = Int32(state0)
    s[state_offset + 1] = Int32(state1)
    s[state_offset + 2] = Int32(state2)
    s[state_offset + 3] = Int32(state3)

    var state4 = Int(s[state_offset + 4])
    var state5 = Int(s[state_offset + 5])
    var state6 = Int(s[state_offset + 6])
    var state7 = Int(s[state_offset + 7])
    for i in range(half):
        tmp0 = Int(s[input_offset + 2 * i])
        var diff = w32(tmp0 - state5)
        diff = w32(diff + 8192) >> 14
        var tmp1 = w32(state4 + w32(diff * resample_allpass(0, 0)))
        state4 = tmp0
        diff = trunc_shift14(w32(tmp1 - state6))
        tmp0 = w32(state5 + w32(diff * resample_allpass(0, 1)))
        state5 = tmp1
        diff = trunc_shift14(w32(tmp0 - state7))
        state7 = w32(state6 + w32(diff * resample_allpass(0, 2)))
        state6 = tmp0
        s[output_offset + 2 * i] = Int32(
            w32(Int(s[output_offset + 2 * i]) + (state7 >> 1)) >> 15
        )
    s[state_offset + 4] = Int32(state4)
    s[state_offset + 5] = Int32(state5)
    s[state_offset + 6] = Int32(state6)
    s[state_offset + 7] = Int32(state7)

    var state8 = Int(s[state_offset + 8])
    var state9 = Int(s[state_offset + 9])
    var state10 = Int(s[state_offset + 10])
    var state11 = Int(s[state_offset + 11])
    for i in range(half):
        tmp0 = Int(s[input_offset + 2 * i])
        var diff = w32(tmp0 - state9)
        diff = w32(diff + 8192) >> 14
        var tmp1 = w32(state8 + w32(diff * resample_allpass(1, 0)))
        state8 = tmp0
        diff = trunc_shift14(w32(tmp1 - state10))
        tmp0 = w32(state9 + w32(diff * resample_allpass(1, 1)))
        state9 = tmp1
        diff = trunc_shift14(w32(tmp0 - state11))
        state11 = w32(state10 + w32(diff * resample_allpass(1, 2)))
        state10 = tmp0
        s[output_offset + 2 * i + 1] = Int32(state11 >> 1)
    s[state_offset + 8] = Int32(state8)
    s[state_offset + 9] = Int32(state9)
    s[state_offset + 10] = Int32(state10)
    s[state_offset + 11] = Int32(state11)

    var state12 = Int(s[state_offset + 12])
    var state13 = Int(s[state_offset + 13])
    var state14 = Int(s[state_offset + 14])
    var state15 = Int(s[state_offset + 15])
    for i in range(half):
        tmp0 = Int(s[input_offset + 2 * i + 1])
        var diff = w32(tmp0 - state13)
        diff = w32(diff + 8192) >> 14
        var tmp1 = w32(state12 + w32(diff * resample_allpass(0, 0)))
        state12 = tmp0
        diff = trunc_shift14(w32(tmp1 - state14))
        tmp0 = w32(state13 + w32(diff * resample_allpass(0, 1)))
        state13 = tmp1
        diff = trunc_shift14(w32(tmp0 - state15))
        state15 = w32(state14 + w32(diff * resample_allpass(0, 2)))
        state14 = tmp0
        s[output_offset + 2 * i + 1] = Int32(
            w32(Int(s[output_offset + 2 * i + 1]) + (state15 >> 1)) >> 15
        )
    s[state_offset + 12] = Int32(state12)
    s[state_offset + 13] = Int32(state13)
    s[state_offset + 14] = Int32(state14)
    s[state_offset + 15] = Int32(state15)


def coefficient_48_32(phase: Int, tap: Int) -> Int:
    if phase == 0:
        if tap == 0: return 778
        if tap == 1: return -2050
        if tap == 2: return 1087
        if tap == 3: return 23285
        if tap == 4: return 12903
        if tap == 5: return -3783
        if tap == 6: return 441
        return 222
    if tap == 0: return 222
    if tap == 1: return 441
    if tap == 2: return -3783
    if tap == 3: return 12903
    if tap == 4: return 23285
    if tap == 5: return 1087
    if tap == 6: return -2050
    return 778


def fractional_48_to_32(s: I32Ptr, input_offset: Int, output_offset: Int, blocks: Int):
    comptime W = simd_width_of[DType.int32]()
    comptime phase0_coefficients = SIMD[DType.int32, W](
        778, -2050, 1087, 23285, 12903, -3783, 441, 222
    )
    comptime phase1_coefficients = SIMD[DType.int32, W](
        222, 441, -3783, 12903, 23285, 1087, -2050, 778
    )
    for block in range(blocks):
        var input_base = input_offset + block * 3
        var output_base = output_offset + block * 2
        var tmp = w32(
            16384 + Int(
                (
                    s.load[width=W](input_base) * phase0_coefficients
                ).reduce_add()
            )
        )
        s[output_base] = Int32(tmp)
        tmp = w32(
            16384 + Int(
                (
                    s.load[width=W](input_base + 1) * phase1_coefficients
                ).reduce_add()
            )
        )
        s[output_base + 1] = Int32(tmp)


def down_int_to_short(
    s: I32Ptr, input_offset: Int, output_offset: Int, state_offset: Int, length: Int
):
    var half = length >> 1
    var state0 = Int(s[state_offset])
    var state1 = Int(s[state_offset + 1])
    var state2 = Int(s[state_offset + 2])
    var state3 = Int(s[state_offset + 3])
    for i in range(half):
        var tmp0 = Int(s[input_offset + 2 * i])
        var diff = w32(tmp0 - state1)
        diff = w32(diff + 8192) >> 14
        var tmp1 = w32(state0 + w32(diff * resample_allpass(1, 0)))
        state0 = tmp0
        diff = trunc_shift14(w32(tmp1 - state2))
        tmp0 = w32(state1 + w32(diff * resample_allpass(1, 1)))
        state1 = tmp1
        diff = trunc_shift14(w32(tmp0 - state3))
        state3 = w32(state2 + w32(diff * resample_allpass(1, 2)))
        state2 = tmp0
        s[input_offset + 2 * i] = Int32(state3 >> 1)
    s[state_offset] = Int32(state0)
    s[state_offset + 1] = Int32(state1)
    s[state_offset + 2] = Int32(state2)
    s[state_offset + 3] = Int32(state3)

    var state4 = Int(s[state_offset + 4])
    var state5 = Int(s[state_offset + 5])
    var state6 = Int(s[state_offset + 6])
    var state7 = Int(s[state_offset + 7])
    for i in range(half):
        var tmp0 = Int(s[input_offset + 2 * i + 1])
        var diff = w32(tmp0 - state5)
        diff = w32(diff + 8192) >> 14
        var tmp1 = w32(state4 + w32(diff * resample_allpass(0, 0)))
        state4 = tmp0
        diff = trunc_shift14(w32(tmp1 - state6))
        tmp0 = w32(state5 + w32(diff * resample_allpass(0, 1)))
        state5 = tmp1
        diff = trunc_shift14(w32(tmp0 - state7))
        state7 = w32(state6 + w32(diff * resample_allpass(0, 2)))
        state6 = tmp0
        s[input_offset + 2 * i + 1] = Int32(state7 >> 1)
    s[state_offset + 4] = Int32(state4)
    s[state_offset + 5] = Int32(state5)
    s[state_offset + 6] = Int32(state6)
    s[state_offset + 7] = Int32(state7)

    var i = 0
    while i < half:
        var first = w32(
            Int(s[input_offset + 2 * i]) + Int(s[input_offset + 2 * i + 1])
        ) >> 15
        s[output_offset + i] = Int32(s16(sat16(first)))
        if i + 1 < half:
            var second = w32(
                Int(s[input_offset + 2 * i + 2]) + Int(s[input_offset + 2 * i + 3])
            ) >> 15
            s[output_offset + i + 1] = Int32(s16(sat16(second)))
        i += 2


def resample_48_to_8(
    s: I32Ptr, input_offset: Int, output_offset: Int, temp_offset: Int
):
    down_short_to_int(s, input_offset, temp_offset + 256, RESAMPLE48, 480)
    lp_int_to_int(s, temp_offset + 256, temp_offset + 16, RESAMPLE48 + 8, 240)
    for i in range(8):
        s[temp_offset + 8 + i] = s[RESAMPLE48 + 24 + i]
        s[RESAMPLE48 + 24 + i] = s[temp_offset + 248 + i]
    fractional_48_to_32(s, temp_offset + 8, temp_offset, 80)
    down_int_to_short(s, temp_offset, output_offset, RESAMPLE48 + 32, 160)


def set_mode(s: I32Ptr, mode: Int) -> Int:
    if mode < 0 or mode > 3:
        return -1
    for i in range(3):
        if mode <= 1:
            s[OVER1 + i] = Int32(8 if i == 0 else (4 if i == 1 else 3))
            s[OVER2 + i] = Int32(14 if i == 0 else (7 if i == 1 else 5))
        else:
            s[OVER1 + i] = Int32(6 if i == 0 else (3 if i == 1 else 2))
            s[OVER2 + i] = Int32(9 if i == 0 else (5 if i == 1 else 3))
    if mode == 0:
        s[INDIVIDUAL] = 24
        s[INDIVIDUAL + 1] = 21
        s[INDIVIDUAL + 2] = 24
        s[TOTAL] = 57
        s[TOTAL + 1] = 48
        s[TOTAL + 2] = 57
    elif mode == 1:
        s[INDIVIDUAL] = 37
        s[INDIVIDUAL + 1] = 32
        s[INDIVIDUAL + 2] = 37
        s[TOTAL] = 100
        s[TOTAL + 1] = 80
        s[TOTAL + 2] = 100
    elif mode == 2:
        s[INDIVIDUAL] = 82
        s[INDIVIDUAL + 1] = 78
        s[INDIVIDUAL + 2] = 82
        s[TOTAL] = 285
        s[TOTAL + 1] = 260
        s[TOTAL + 2] = 285
    else:
        for i in range(3):
            s[INDIVIDUAL + i] = 94
        s[TOTAL] = 1100
        s[TOTAL + 1] = 1050
        s[TOTAL + 2] = 1100
    return 0


def initialize(s: I32Ptr):
    for i in range(4096):
        s[i] = 0
    for i in range(12):
        s[NOISE_MEANS + i] = Int32(noise_mean_initial(i))
        s[SPEECH_MEANS + i] = Int32(speech_mean_initial(i))
        s[NOISE_STDS + i] = Int32(noise_std_initial(i))
        s[SPEECH_STDS + i] = Int32(speech_std_initial(i))
    for i in range(96):
        s[LOW_VALUES + i] = 10000
    for i in range(6):
        s[MEAN_VALUE + i] = 1600
    _ = set_mode(s, 0)


def valid_rate_length(rate: Int, length: Int) -> Bool:
    if rate != 8000 and rate != 16000 and rate != 32000 and rate != 48000:
        return False
    return length == rate / 100 or length == rate / 50 or length == (rate * 3) / 100


@export("mwv_state_length")
def mwv_state_length() abi("C") -> Int:
    return 4096


@export("mwv_init")
def mwv_init(state_address: Int) abi("C") -> Int:
    if state_address == 0:
        return -1
    initialize(I32Ptr(unsafe_from_address=state_address))
    return 0


@export("mwv_set_mode")
def mwv_set_mode(state_address: Int, mode: Int) abi("C") -> Int:
    if state_address == 0:
        return -1
    return set_mode(I32Ptr(unsafe_from_address=state_address), mode)


@export("mwv_valid_rate_and_frame_length")
def mwv_valid_rate_and_frame_length(rate: Int, length: Int) abi("C") -> Int:
    return 1 if valid_rate_length(rate, length) else 0


def process_frame(state_address: Int, audio_address: Int, rate: Int, length: Int) -> Int:
    if state_address == 0 or audio_address == 0 or not valid_rate_length(rate, length):
        return -1
    var s = I32Ptr(unsafe_from_address=state_address)
    var audio = I16Ptr(unsafe_from_address=audio_address)
    comptime W = simd_width_of[DType.int16]()
    var i = 0
    while i + W <= length:
        var samples = audio.load[width=W](i).cast[DType.int32]()
        s.store(TEMP32 + i, samples)
        i += W
    while i < length:
        s[TEMP32 + i] = Int32(audio[i])
        i += 1
    var vad_length = length
    if rate == 8000:
        comptime COPY_W = simd_width_of[DType.int32]()
        i = 0
        while i + COPY_W <= length:
            s.store(DATA8 + i, s.load[width=COPY_W](TEMP32 + i))
            i += COPY_W
        while i < length:
            s[DATA8 + i] = s[TEMP32 + i]
            i += 1
    elif rate == 16000:
        downsample_by_2(s, TEMP32, DATA8, DOWNSAMPLE, length)
        vad_length = length >> 1
    elif rate == 32000:
        downsample_by_2(s, TEMP32, TEMP32 + 960, DOWNSAMPLE + 2, length)
        downsample_by_2(s, TEMP32 + 960, DATA8, DOWNSAMPLE, length >> 1)
        vad_length = length >> 2
    else:
        var frames = length / 480
        # Preserve WebRTC VAD 2.0.10's fixed input pointer across 10 ms blocks.
        for frame in range(frames):
            resample_48_to_8(
                s, TEMP32, DATA8 + frame * 80, 2600
            )
        vad_length = length / 6
    var total_power = calculate_features(s, vad_length)
    return 1 if gmm_probability(s, total_power, vad_length) > 0 else 0


@export("mwv_process")
def mwv_process(state_address: Int, audio_address: Int, rate: Int, length: Int) abi("C") -> Int:
    return process_frame(state_address, audio_address, rate, length)


@export("mwv_process_many")
def mwv_process_many(
    state_address: Int, audio_address: Int, result_address: Int,
    rate: Int, frame_length: Int, frame_count: Int
) abi("C") -> Int:
    if (
        state_address == 0 or frame_count < 0
        or not valid_rate_length(rate, frame_length)
        or (frame_count > 0 and (audio_address == 0 or result_address == 0))
    ):
        return -1
    if frame_count == 0:
        return 0
    var results = U8Ptr(unsafe_from_address=result_address)
    for frame in range(frame_count):
        var decision = process_frame(
            state_address, audio_address + frame * frame_length * 2,
            rate, frame_length
        )
        if decision < 0:
            return -1
        results[frame] = UInt8(decision)
    return 0
