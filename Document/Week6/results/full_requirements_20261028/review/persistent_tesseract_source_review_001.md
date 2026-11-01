# PersistentTesseract 独立只读源码审查 001

审查者：`/root/week6_replanning_review`，2026-10-28。
范围：`src/gui_agent/perception/tesseract_api.py` 及 `ocr.py` 的显式transport分支。
未修改这些源码，未执行OCR、GUI、模型或性能测试。实现仍待owner最终冻结，
本记录不替代正式新留出数值与性能验收。

结论：本轮读到的 ABI、所有权和串行化路径未发现发布阻塞项。正式接受仍需
冻结DLL、语言资源和评分源码，并通过预登记的完整输出等同/公平冷暖性能门槛。

- C ABI 的 opaque handle、输入buffer和返回字符串使用 `c_void_p`；标量与
  enum使用 `c_int`，UTF8配置串使用 `c_char_p`，无 stdcall 错配。已对照
  [Tesseract 5.5.1 capi.h](https://raw.githubusercontent.com/tesseract-ocr/tesseract/5.5.1/include/tesseract/capi.h)。
- `GetTsvText` 返回地址先经 `string_at`复制并解码，再在finally中调用
  `TessDeleteText`；没有把ctypes自动转换的临时Python bytes当原生地址释放。
  API实例在配置key变化、close或弱引用终结时 `End` 后 `Delete`，finalizer
  保有library对象和handle引用。显式close有幂等终态保护，失败初始化也释放handle。
- 连续uint8 L/RGB/RGBA数组保留到识别和取字结束，真实stride、宽高、通道数传入；
  非8bit模式被拒。`SetImage`、每帧 `ClearAdaptiveClassifier` 与finally `Clear`
  的顺序符合页间重置用途。官方接口说明 `SetImage` 自行复制像素、`Clear`移除
  当前图像与结果而保留昂贵识别数据；这支持生命周期判断，不证明所有图像数值等同。
  参见 [baseapi.h](https://raw.githubusercontent.com/tesseract-ocr/tesseract/5.5.1/include/tesseract/baseapi.h)。
- 单实例 `RLock` 覆盖初始化、像素提交、识别、取字、清理及close。适配器不调用
  全局 `SetVariable`；没有发现单实例并发SetImage或Clear引起的使用后释放通路。
- DLL缺失、初始化失败、Recognize失败、空指针、损坏TSV及解码异常向外显式报错；
  上层转换为 `OcrError`，没有悄悄退回subprocess使速度口径失真。
- TSV头由CLI renderer添加而非GetTSVText返回；实现显式补列名并按已有
  pytesseract字典路径将数值转为 `int(float(...))`，避免小数置信度改变过滤排序。
  对照 [renderer.cpp](https://raw.githubusercontent.com/tesseract-ocr/tesseract/5.5.1/src/api/renderer.cpp)。

非阻塞边界：`TesseractOCREngine.close`只释放当前已创建transport，不承诺未初始化
engine永久关闭；若与首次recognize并发，可能先看见None而错过刚创建实例。底层
transport仍有RLock和终结器，未见由此导致重复释放。owner已明确该API只是释放
当前实例，底层 `PersistentTesseract.close` 自身则是不可再用的终态。

数值与计时限制：自适应清理、页状态重置和开发帧相等不能推出普遍OCR等价；
PNG分辨率元数据、语言资源、实际加载DLL与基线版本仍需绑定。本轮审查不称
所有场景安全、不称速度达标，也不以结构合理替代正式两批真实CPU测量。

