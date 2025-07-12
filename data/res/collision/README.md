# 实验数据参数解释

|               编号               | sample_per_length | max_length | 双方层数是否透明 |   sample_per_length终止条件？   | 最大drop数 |              备注              |
| :------------------------------: | :---------------: | :--------: | :--------------: | :------------------------------: | :--------: | :----------------------------: |
|    collision_char_10_0_v1.pkl    |        20        |    100    |        否        | success+fail = sample_per_length |     20     |                                |
|    collision_char_10_0_v2.pkl    |        50        |    100    |        是        | success+fail = sample_per_length |     10     |                                |
|    collision_char_10_0_v3.pkl    |        20        |    100    |        是        | success+fail = sample_per_length |     10     |                                |
|    collision_char_10_0_v4.pkl    |        70        |    100    |        是        | success+fail = sample_per_length |     10     |      由v2,v3加权合并得到      |
|    collision_char_10_0_v5.pkl    |        70        |     64     |        否        | success+fail = sample_per_length |     20     |                                |
|    collision_char_25_0_v1.pkl    |        20        |    100    |        否        | success+fail = sample_per_length |     10     |                                |
|    collision_char_15_0_v1.pkl    |        50        |     64     |        是        | success+fail = sample_per_length |     10     |                                |
|    collision_char_15_0_v2.pkl    |        70        |     64     |        否        | success+fail = sample_per_length |     20     |                                |
|    collision_char_25_0_v1.pkl    |        70        |     64     |        是        | success+fail = sample_per_length |     10     |                                |
|      collision_char_v0.pkl      |        20        |    100    |        是        | success+fail = sample_per_length |     10     |                                |
|      collision_char_v1.pkl      |        10        |     64     |        是        | success+fail = sample_per_length |     10     |                                |
|      collision_char_v2.pkl      |        10        |     64     |        是        | success+fail = sample_per_length |     10     |                                |
|      collision_char_v3.pkl      |        90        |     64     |        是        | success+fail = sample_per_length |     10     | 由v0、v1、v2、base加权合并得到 |
|        base_collision.pkl        |        50        |    100    |        是        |   success = sample_per_length   |     无     |                                |
|    collision_char_20_0_v0.pkl    |        70        |     64     |        是        | success+fail = sample_per_length |     10     |                                |
| collision_char_Math500_100_0_v1 |        70        |     64     |        否        | success+fail = sample_per_length |     20     |                                |
| collision_char_Math500_1000_0_v1 |        100        |     64     |        否        | success+fail = sample_per_length |     20     |                                |
