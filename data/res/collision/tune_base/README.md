# 实验数据参数解释

| 编号                       | sample_per_length | max_length | 双方层数是否透明 | sample_per_length终止条件？      | 最大drop数 | 备注                |
| -------------------------- | :---------------: | :--------: | :--------------: | -------------------------------- | :--------: | ------------------- |
| collision_char_10_0_v1.pkl |        20        |    100    |        否        | success+fail = sample_per_length |     20     |                     |
| collision_char_10_0_v2.pkl |        50        |    100    |        是        | success+fail = sample_per_length |     10     |                     |
| collision_char_10_0_v3.pkl |        20        |    100    |        是        | success+fail = sample_per_length |     10     |                     |
| collision_char_25_0_v1.pkl |        20        |    100    |        否        | success+fail = sample_per_length |     10     |                     |
| collision_char_10_0_v4.pkl |        70        |    100    |        是        | success+fail = sample_per_length |     10     | 由v2,v3加权合并得到 |
| collision_char_15_0_v1.pkl |        50        |     64     |        是        | success+fail = sample_per_length |     10     |                     |
|                            |                  |            |                  |                                  |            |                     |
|                            |                  |            |                  |                                  |            |                     |
|                            |                  |            |                  |                                  |            |                     |
