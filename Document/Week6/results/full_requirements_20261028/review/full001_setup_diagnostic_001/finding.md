# Full validation 001 的 setup 环境诊断

执行者：`/root/week6_replanning_review`。仅按Root授权运行一个新独立测试节点，
没有修改生产/测试/脚本/配置，没有修复或删除原full001临时目录，没有中止进程。

原harness指定 `Intern/tmp/full_validation_001/pytest` 作为pytest的 `--basetemp`，
但其父目录 `Intern/tmp/full_validation_001` 不存在。保存到实验记录的目录在
另一个 `Document/Week6/results/...` 路径，创建它不会创建测试临时目录的父目录。

单节点复现使用另一个不存在父目录的同类路径，运行
`tests/test_config.py::test_an_unknown_top_level_key_is_rejected`。实际pytest退出1，
`1 error in 0.44s`，错误发生在setup，尚未进入测试函数中的配置读取断言。
原始trace显示 `WindowsPath.mkdir(..., parents=False)` 触发
`FileNotFoundError: [WinError 3]`；不是验证器返回了错误结论，也不是模型/GUI失败。

原始证据：stdout SHA256
`6237c7746475e5f44e3b1e6268bbc50f6a2ef3abb022ed0ec5c835f2d7aebccc`；
实际子退出与前后源码pins审计 SHA256
`91e816dc2ee740b124bf3efa5e5223bda22d43c7e471326da4d3cf6ede06277d`。
审计中所有所选代码pins前后相同。原full001的目录、输出和进程均保留。

可采取的修复只涉及未来私有harness：在全新full002启动pytest前先创建其唯一
basetemp父目录，或将全新basetemp放在已存在父目录下。不能改原full001输出，
也不能据此把原full001记为通过。完成后仍需一次真实的新完整回归。

这条证据解释使用 `tmp_path` 的大量setup E，不自动解释同时出现的普通F或
其他问题；那些错误需等原full001完整traceback后分别判断，不盲改生产来消红。

