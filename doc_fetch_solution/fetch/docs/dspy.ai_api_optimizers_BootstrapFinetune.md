# Bootstrapfinetune

**Source:** https://dspy.ai/api/optimizers/BootstrapFinetune
**Fetched:** 2025-08-24 17:10:37
**Status:** Success

---

[

](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/api/optimizers/BootstrapFinetune.md)
# dspy.BootstrapFinetune
## 
 dspy.BootstrapFinetune(metric: Callable | None = None, multitask: bool = True, train_kwargs: dict[str, Any] | dict[LM, dict[str, Any]] | None = None, adapter: Adapter | dict[LM, Adapter] | None = None, exclude_demos: bool = False, num_threads: int | None = None)
Bases: FinetuneTeleprompter
Source code in
dspy/teleprompt/bootstrap_finetune.py
```
[40](#__codelineno-0-40)
[41](#__codelineno-0-41)
[42](#__codelineno-0-42)
[43](#__codelineno-0-43)
[44](#__codelineno-0-44)
[45](#__codelineno-0-45)
[46](#__codelineno-0-46)
[47](#__codelineno-0-47)
[48](#__codelineno-0-48)
[49](#__codelineno-0-49)
[50](#__codelineno-0-50)
[51](#__codelineno-0-51)
[52](#__codelineno-0-52)
[53](#__codelineno-0-53)
[54](#__codelineno-0-54)
[55](#__codelineno-0-55)
[56](#__codelineno-0-56)
[57](#__codelineno-0-57)
[58](#__codelineno-0-58)
[59](#__codelineno-0-59)
[60](#__codelineno-0-60)
[61](#__codelineno-0-61)
```
```
def __init__(
    self,
    metric: Callable | None = None,
    multitask: bool = True,
    train_kwargs: dict[str, Any] | dict[LM, dict[str, Any]] | None = None,
    adapter: Adapter | dict[LM, Adapter] | None = None,
    exclude_demos: bool = False,
    num_threads: int | None = None,
):
    # TODO(feature): Inputs train_kwargs (a dict with string keys) and
    # adapter (Adapter) can depend on the LM they are used with. We are
    # takingthese as parameters for the time being. However, they can be
    # attached to LMs themselves -- an LM could know which adapter it should
    # be used with along with the train_kwargs. This will lead the only
    # required argument for LM.finetune() to be the train dataset.

    super().__init__(train_kwargs=train_kwargs)
    self.metric = metric
    self.multitask = multitask
    self.adapter: dict[LM, Adapter] = self.convert_to_lm_dict(adapter)
    self.exclude_demos = exclude_demos
    self.num_threads = num_threads

```
### Functions
#### 
 compile(student: Module, trainset: list[Example], teacher: Module | list[Module] | None = None) -> Module
Source code in
dspy/teleprompt/bootstrap_finetune.py
```
[ 63](#__codelineno-0-63)
[ 64](#__codelineno-0-64)
[ 65](#__codelineno-0-65)
[ 66](#__codelineno-0-66)
[ 67](#__codelineno-0-67)
[ 68](#__codelineno-0-68)
[ 69](#__codelineno-0-69)
[ 70](#__codelineno-0-70)
[ 71](#__codelineno-0-71)
[ 72](#__codelineno-0-72)
[ 73](#__codelineno-0-73)
[ 74](#__codelineno-0-74)
[ 75](#__codelineno-0-75)
[ 76](#__codelineno-0-76)
[ 77](#__codelineno-0-77)
[ 78](#__codelineno-0-78)
[ 79](#__codelineno-0-79)
[ 80](#__codelineno-0-80)
[ 81](#__codelineno-0-81)
[ 82](#__codelineno-0-82)
[ 83](#__codelineno-0-83)
[ 84](#__codelineno-0-84)
[ 85](#__codelineno-0-85)
[ 86](#__codelineno-0-86)
[ 87](#__codelineno-0-87)
[ 88](#__codelineno-0-88)
[ 89](#__codelineno-0-89)
[ 90](#__codelineno-0-90)
[ 91](#__codelineno-0-91)
[ 92](#__codelineno-0-92)
[ 93](#__codelineno-0-93)
[ 94](#__codelineno-0-94)
[ 95](#__codelineno-0-95)
[ 96](#__codelineno-0-96)
[ 97](#__codelineno-0-97)
[ 98](#__codelineno-0-98)
[ 99](#__codelineno-0-99)
[100](#__codelineno-0-100)
[101](#__codelineno-0-101)
[102](#__codelineno-0-102)
[103](#__codelineno-0-103)
[104](#__codelineno-0-104)
[105](#__codelineno-0-105)
[106](#__codelineno-0-106)
[107](#__codelineno-0-107)
[108](#__codelineno-0-108)
[109](#__codelineno-0-109)
[110](#__codelineno-0-110)
[111](#__codelineno-0-111)
[112](#__codelineno-0-112)
[113](#__codelineno-0-113)
[114](#__codelineno-0-114)
[115](#__codelineno-0-115)
[116](#__codelineno-0-116)
[117](#__codelineno-0-117)
[118](#__codelineno-0-118)
[119](#__codelineno-0-119)
[120](#__codelineno-0-120)
[121](#__codelineno-0-121)
[122](#__codelineno-0-122)
[123](#__codelineno-0-123)
[124](#__codelineno-0-124)
[125](#__codelineno-0-125)
[126](#__codelineno-0-126)
[127](#__codelineno-0-127)
[128](#__codelineno-0-128)
[129](#__codelineno-0-129)
[130](#__codelineno-0-130)
[131](#__codelineno-0-131)
[132](#__codelineno-0-132)
[133](#__codelineno-0-133)
[134](#__codelineno-0-134)
[135](#__codelineno-0-135)
[136](#__codelineno-0-136)
[137](#__codelineno-0-137)
```
```
def compile(
    self, student: Module, trainset: list[Example], teacher: Module | list[Module] | None = None
) -> Module:
    # TODO: Print statements can be converted to logger.info if we ensure
    # that the default DSPy logger logs info level messages in notebook
    # environments.
    logger.info("Preparing the student and teacher programs...")
    all_predictors_have_lms(student)

    logger.info("Bootstrapping data...")
    trace_data = []

    teachers = teacher if isinstance(teacher, list) else [teacher]
    teachers = [prepare_teacher(student, t) for t in teachers]
    num_threads = self.num_threads or dspy.settings.num_threads
    for t in teachers:
        trace_data += bootstrap_trace_data(program=t, dataset=trainset, metric=self.metric, num_threads=num_threads)

    logger.info("Preparing the train data...")
    key_to_data = {}
    for pred_ind, pred in enumerate(student.predictors()):
        data_pred_ind = None if self.multitask else pred_ind
        if pred.lm is None:
            raise ValueError(
                f"Predictor {pred_ind} does not have an LM assigned. "
                f"Please ensure the module's predictors have their LM set before fine-tuning. "
                f"You can set it using: your_module.set_lm(your_lm)"
            )
        training_key = (pred.lm, data_pred_ind)

        if training_key not in key_to_data:
            train_data, data_format = self._prepare_finetune_data(
                trace_data=trace_data, lm=pred.lm, pred_ind=data_pred_ind
            )
            logger.info(f"Using {len(train_data)} data points for fine-tuning the model: {pred.lm.model}")
            finetune_kwargs = {
                "lm": pred.lm,
                "train_data": train_data,
                "train_data_format": data_format,
                "train_kwargs": self.train_kwargs[pred.lm],
            }
            key_to_data[training_key] = finetune_kwargs

    logger.info("Starting LM fine-tuning...")
    # TODO(feature): We could run batches of fine-tuning jobs in sequence
    # to avoid exceeding the number of threads.
    if len(key_to_data) > num_threads:
        raise ValueError(
            "BootstrapFinetune requires `num_threads` to be bigger than or equal to the number of fine-tuning "
            f"jobs. There are {len(key_to_data)} fine-tuning jobs to start, but the number of threads is: "
            f"{num_threads}! If the `multitask` flag is set to False, the number of fine-tuning jobs will "
            "be equal to the number of predictors in the student program. If the `multitask` flag is set to True, "
            "the number of fine-tuning jobs will be equal to: 1 if there is only a context LM, or the number of "
            "unique LMs attached to the predictors in the student program. In any case, the number of fine-tuning "
            "jobs will be less than or equal to the number of predictors."
        )
    logger.info(f"{len(key_to_data)} fine-tuning job(s) to start")
    key_to_lm = self.finetune_lms(key_to_data)

    logger.info("Updating the student program with the fine-tuned LMs...")
    for pred_ind, pred in enumerate(student.predictors()):
        data_pred_ind = None if self.multitask else pred_ind
        training_key = (pred.lm, data_pred_ind)
        finetuned_lm = key_to_lm[training_key]
        if isinstance(finetuned_lm, Exception):
            raise RuntimeError(f"Finetuned LM for predictor {pred_ind} failed.") from finetuned_lm
        pred.lm = finetuned_lm
        # TODO: What should the correct behavior be here? Should
        # BootstrapFinetune modify the prompt demos according to the
        # train data?
        pred.demos = [] if self.exclude_demos else pred.demos

    logger.info("BootstrapFinetune has finished compiling the student program")
    student._compiled = True
    return student

```
#### 
 convert_to_lm_dict(arg) -> dict[LM, Any]

staticmethod
Source code in
dspy/teleprompt/bootstrap_finetune.py
```
[30](#__codelineno-0-30)
[31](#__codelineno-0-31)
[32](#__codelineno-0-32)
[33](#__codelineno-0-33)
[34](#__codelineno-0-34)
[35](#__codelineno-0-35)
[36](#__codelineno-0-36)
```
```
@staticmethod
def convert_to_lm_dict(arg) -> dict[LM, Any]:
    non_empty_dict = arg and isinstance(arg, dict)
    if non_empty_dict and all(isinstance(k, LM) for k in arg.keys()):
        return arg
    # Default to using the same value for all LMs
    return defaultdict(lambda: arg)

```
#### 
 finetune_lms(finetune_dict) -> dict[Any, LM]

staticmethod
Source code in
dspy/teleprompt/bootstrap_finetune.py
```
[139](#__codelineno-0-139)
[140](#__codelineno-0-140)
[141](#__codelineno-0-141)
[142](#__codelineno-0-142)
[143](#__codelineno-0-143)
[144](#__codelineno-0-144)
[145](#__codelineno-0-145)
[146](#__codelineno-0-146)
[147](#__codelineno-0-147)
[148](#__codelineno-0-148)
[149](#__codelineno-0-149)
[150](#__codelineno-0-150)
[151](#__codelineno-0-151)
[152](#__codelineno-0-152)
[153](#__codelineno-0-153)
[154](#__codelineno-0-154)
[155](#__codelineno-0-155)
[156](#__codelineno-0-156)
[157](#__codelineno-0-157)
[158](#__codelineno-0-158)
[159](#__codelineno-0-159)
[160](#__codelineno-0-160)
[161](#__codelineno-0-161)
[162](#__codelineno-0-162)
[163](#__codelineno-0-163)
[164](#__codelineno-0-164)
[165](#__codelineno-0-165)
[166](#__codelineno-0-166)
[167](#__codelineno-0-167)
[168](#__codelineno-0-168)
[169](#__codelineno-0-169)
```
```
@staticmethod
def finetune_lms(finetune_dict) -> dict[Any, LM]:
    num_jobs = len(finetune_dict)
    logger.info(f"Starting {num_jobs} fine-tuning job(s)...")
    # TODO(nit) Pass an identifier to the job so that we can tell the logs
    # coming from different fine-tune threads.

    key_to_job = {}
    for key, finetune_kwargs in finetune_dict.items():
        lm: LM = finetune_kwargs.pop("lm")
        # TODO: The following line is a hack. We should re-think how to free
        # up resources for fine-tuning. This might mean introducing a new
        # provider method (e.g. prepare_for_finetune) that can be called
        # before fine-tuning is started.
        logger.info(
            "Calling lm.kill() on the LM to be fine-tuned to free up resources. This won't have any effect if the "
            "LM is not running."
        )
        lm.kill()
        key_to_job[key] = lm.finetune(**finetune_kwargs)

    key_to_lm = {}
    for ind, (key, job) in enumerate(key_to_job.items()):
        result = job.result()
        if isinstance(result, Exception):
            raise result
        key_to_lm[key] = result
        job.thread.join()
        logger.info(f"Job {ind + 1}/{num_jobs} is done")

    return key_to_lm

```
#### 
 get_params() -> dict[str, Any]
Get the parameters of the teleprompter.
Returns:
Type
Description
dict
[
str
,
Any
]
The parameters of the teleprompter.
Source code in
dspy/teleprompt/teleprompt.py
```
[25](#__codelineno-0-25)
[26](#__codelineno-0-26)
[27](#__codelineno-0-27)
[28](#__codelineno-0-28)
[29](#__codelineno-0-29)
[30](#__codelineno-0-30)
[31](#__codelineno-0-31)
[32](#__codelineno-0-32)
```
```
def get_params(self) -> dict[str, Any]:
    """
    Get the parameters of the teleprompter.

    Returns:
        The parameters of the teleprompter.
    """
    return self.__dict__

```
:::