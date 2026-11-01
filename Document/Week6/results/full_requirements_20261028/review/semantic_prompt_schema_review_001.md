# 语义请求与严格schema只读复核 001

审查者：`/root/week6_replanning_review`。范围仅生产提示词、实际已保存的
normal_002首个请求与protocol；未追加模型请求、未改源码，未读取活动fixture隐藏状态。
本记录在该HTTP响应返回前形成，不猜测尚未返回的原始内容。

绑定SHA256：

- `src/gui_agent/planning/semantic_workflow.py`：
  `885178ca770c28177afaec724e2ebeff5abb238a234683bd06f88183f8e041fb`
- `live/normal_002/run/automatic/semantic_plan_round_001/attempt_01_request.json`：
  `0e47b5939d10337902c7040b1991d47531eb2cd6b0fc47331cd31c151135939a`
- 同目录 `protocol.json`：
  `26d5d2199e8b52409d390540025ff59dac0399d805a2898d8f0595f7cf3dfa42`

未发现必须阻塞的JSON结构矛盾。root和goal对象都禁止额外字段，所有已定义属性
都在required中，`$ref`引用存在。schema允许空目标，随后由本地状态校验要求
ready初始至少两个、非ready必须零目标；这是额外的失败关闭校验，不是相反规则。
属性序与required数组序不同不等于对象不可生成。

提示词有能力上的张力：要求一次给出未来有序方案，同时不能把未见未来文字
说成已观察到。但请求本身提供了 `Project ALPHA`、`Theme GREEN`、`LOCAL COMPLETE`
这些最终文字，初屏仅显示项目选择。按不同语义阶段使用用户给出的未满足后态、
把审阅与本地提交合为一个有意义末目标，仍有可能构成可检查序列；不需要要求
四个当前不可见的未来界面标题。因此现有文字约束不能据此断言为逻辑无解。

长输出风险确实存在：reason允许4000字符，每个instruction允许4000字符，
最多8目标，每目标正负检查各最多16串，且提示词没有要求简短。这些是宽松上限，
不是必须写满的长度。上下文还包含完整OCR/候选几何及额外通知区文字，可能增加
理解和选择负担；这是实际全屏观察范围，不是提供给模型的fixture隐藏状态。

normal_001的已知记录由Root报告为1规划/1实际HTTP、0输入、0unknown，240秒API
超时后真实子exit1；该事实只能归为传输等待超时/未完成方案。当前审查没有原始
完整成功回复可据，不能归因为GUI失败、模型质量零分或schema必然不能满足。
Root已用独立新尝试延长显式等待与全局上限，生产检查未降低。应等normal_002
真实回复返回后，再按完整原文判断冗长、重复、截断、拒绝或可执行方案，保留失败。

