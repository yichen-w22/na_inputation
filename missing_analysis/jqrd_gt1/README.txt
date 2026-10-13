JQRD GT1 每日记录间隔分布共享聚类

在项目根目录运行（Python 3.13，依赖精确版本见 requirements.txt）：
  python -m unittest discover -s missing_analysis -p test_daily_modes.py -v
  python missing_analysis/daily_modes.py prepare
  python missing_analysis/daily_modes.py evaluate
  python missing_analysis/daily_modes.py report
  python missing_analysis/daily_modes.py verify

默认输出：missing_analysis/runs/jqrd_gt1_daily_v1
每项命令均支持 --run 路径；协议参数变化必须用新的运行目录。
prepare 可从完整测点缓存恢复，evaluate 可从已完成的K恢复。
未完成的K重跑；源文件改变时禁止沿用缓存。

人工确认后选择（将N改为2、3、4、5或6）：
  python missing_analysis/daily_modes.py select --k N
该命令不扫描原始CSV、不重新拟合，只切换主结果并更新摘要图与报告。
默认只是规则推荐，状态 pending_human_confirmation；未代替用户确认。

入口：report.txt（结论），main_result.json（主结果索引），k_comparison.csv。
所有图为Matplotlib SVG；所有数值结果、代码快照、种子及来源元数据保留。
daily_profiles.npz 按 daily_metadata.csv 的 sample_id 排列。
52维依次为 <0.01、50个对数箱、>1000；1000包含在第50个常规箱。
无完整间隔时概率为NaN；不以全零向量参与聚类。
聚类标签CSV从1起；内部NPZ标签从0起，-1表示未分类。
边界为独立布尔列；stable / unstable / insufficient_evidence 不是额外类别。
每个K内类别按代表分布中位箱排序；跨K编号不自动对应。
每日计数归间隔左端，长间隔覆盖时长归实际重叠的每个日期。
月度切换计入后一个日期所属月份；比例分母为可分类日期。
连续段输出两端删失标记，不把可见长度称为完整机制持续时间。
各K的转移、持续时间、500次随机打乱均计算；转移及持续图只生成当前主结果。

环境、代码哈希和命令记录在 provenance_*.json；源CSV验证为大小及修改时间核对，
不是全文件内容哈希。来源选取表另有SHA256；源数据只读。
不筛运行状态、不解释故障原因、不据此声称填补精度或独立外部验证。
