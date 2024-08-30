# 2. Forgetting Score를 활용한 도로 표지판 분류 모델 구현

## 1. 개요

### **1. 프로젝트 배경**

 도로 표지판 인식은 자율 주행 시스템과 교통 관리 시스템에서 중요한 역할을 합니다. 정확한 도로 표지판 인식을 위해서는 다양한 환경과 조건에서 견고하게 작동할 수 있는 딥러닝 모델이 필요합니다. 하지만 모든 학습 데이터가 모델 성능에 긍정적인 영향을 미치는 것은 아닙니다. 이 프로젝트는 데이터셋 내의 특정 예제들이 모델 성능에 미치는 영향을 분석하기 위해 Forgetting Score를 도입하고, 이를 활용하여 도로 표지판 분류 모델의 성능을 최적화하는 것을 목표로 합니다.

### **2. 프로젝트 목적**

 이 프로젝트의 목적은 도로 표지판을 정확하게 분류하는 딥러닝 모델을 개발하는 것입니다. 특히 Forgetting Score를 활용하여 모델 학습에 도움이 되지 않거나, 오히려 성능을 저하시킬 수 있는 데이터를 식별하고 제거함으로써 모델의 일반화 성능을 향상시키는 것을 목표로 합니다. 최종적으로, 이 모델은 다양한 환경에서 일관되게 높은 성능을 발휘할 수 있도록 설계됩니다.

### **3. 접근 방식**

- **데이터 분석 및 전처리**: Kaggle에서 도로 표지판 데이터를 수집한 후, 데이터를 탐색하고 전처리합니다. 이 과정에서 이미지를 일관된 크기로 조정하고, 필요한 경우 데이터를 정규화하여 모델 입력에 적합한 형식으로 변환합니다.
- **CIFAR-10 실험**: 먼저 CIFAR-10 데이터셋을 사용하여 실험을 진행하였습니다. CIFAR-10 데이터셋은 10개의 클래스로 구성된 작은 이미지 데이터셋으로, 모델의 초기 성능을 평가하고, Forgetting Score 기반의 데이터 셋 최적화 전략을 테스트하는 데 적합합니다. CIFAR-10 데이터에 대해 이미지를 일관된 크기로 조정하고, 데이터를 정규화하여 모델 입력에 적합한 형식으로 변환합니다.
- **모델 선택 및 학습**: ResNet18 모델을 선택하여 학습을 진행합니다. 초기 학습 후, Forgetting Score를 계산하여 데이터셋에서 성능에 부정적인 영향을 미치는 예제들을 식별합니다.
- **데이터셋 최적화**: Forgetting Score를 기준으로 일부 데이터를 제거하거나 유지하여 데이터셋을 최적화합니다. 이후, 최적화된 데이터셋을 사용하여 모델을 재학습하고, 성능을 평가합니다.
- **모델 평가**: 최종 모델의 성능은 정확도(Accuracy), 혼동 행렬(Confusion Matrix) 등 다양한 평가 지표를 통해 측정되며, 최적화 이전과 이후의 성능을 비교 분석합니다.

### **4. 사용 기술 및 도구**

- **프로그래밍 언어**: Python
- **딥러닝 프레임워크**: PyTorch
- **데이터 분석 도구**: Pandas, NumPy
- **시각화 도구**: Matplotlib, Seaborn
- **모델**: ResNet18 (사전 학습된 모델을 사용하여 초기화)
- **기타**: Forgetting Score 계산을 위한 커스텀 코드

### **5. 기대 효과 및 비즈니스 임팩트**

 이 프로젝트는 도로 표지판 분류 모델의 성능을 최적화하는 과정에서, 불필요하거나 오히려 성능을 저해하는 데이터를 효과적으로 제거할 수 있음을 입증합니다. 이는 자율 주행 차량 및 교통 관리 시스템에서 도로 표지판 인식의 정확도를 높이는 데 기여할 수 있습니다. 더불어, 이 접근 방식은 다른 이미지 분류 문제에도 적용될 수 있어, 모델의 효율성을 높이고 불필요한 데이터의 처리 비용을 절감하는 데 도움이 될 수 있습니다. 결과적으로, 이 프로젝트는 도로 안전성을 향상시키고, 자율 주행 기술의 발전에 기여할 수 있는 중요한 성과를 제공합니다.

## 2. 실험 내용

### 1. Cifar10 실험

1. **개요**
    
    CIFAR-10 데이터셋을 활용하여 ResNet18 모델을 학습하고, Forgetting Score를 측정하여 데이터셋 내 예제의 학습 영향도를 평가합니다. 이를 통해 데이터셋에서 학습 성능에 부정적인 영향을 미치는 예제를 제거하거나 유지하는 전략을 적용하여 모델 성능을 최적화합니다.
    
2. **실험과정**
    - **데이터셋 구성**: CIFAR-10은 10개의 클래스로 구성된 이미지 데이터셋으로, 각 클래스는 6000개의 이미지로 이루어져 있습니다. 이 데이터셋을 사용하여 이미지 분류 모델을 학습합니다.
    - **데이터 전처리**: 데이터를 정규화하여 모델 입력에 적합한 형식으로 변환합니다. 이 과정에서 데이터 증강 기법을 사용하여 모델의 일반화 성능을 향상시킵니다.
    - **모델 학습**: ResNet18 모델을 사용하여 CIFAR-10 데이터셋에 대해 학습을 진행합니다. 학습 과정에서 모델의 성능을 모니터링하며, 각 예제에 대한 Forgetting Score를 계산하여 학습에 대한 기여도를 평가합니다.
    - **데이터셋 최적화**: Forgetting Score를 기준으로 데이터셋을 최적화합니다. 학습에 부정적인 영향을 미치는 예제를 제거하거나 유지하는 전략을 적용한 후, 모델을 재학습하여 성능 변화를 분석합니다.
    - **모델 평가**: 최종 모델의 성능을 정확도(Accuracy)와 같은 지표로 평가하며, 최적화 이전과 이후의 성능을 비교 분석합니다.
    
3. **코드 분석**
    
    **<데이터 전처리 및 로드>**
    
    ```python
    normalize = transforms.Normalize(
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])
    
    train_dataset = datasets.CIFAR10(
        root='data_path', train=True, transform=train_transform, download=True)
    
    test_dataset = datasets.CIFAR10(
        root='data_path', train=False, transform=test_transform, download=True)
    ```
    
        **설명**
    
    - **데이터 전처리**: 학습 데이터를 위한 증강 기법으로 `RandomCrop(32, padding=4)`과 `RandomHorizontalFlip()`을 사용합니다. 모든 데이터는 `ToTensor`로 변환되며, 정규화를 적용합니다.
    - **데이터 로드**: CIFAR-10 데이터셋을 학습용과 테스트용으로 로드하며, 각 데이터셋에 지정된 전처리를 적용합니다.
    
    **<모델 학습 (Train Loop)>**
    
    ```python
    def train(args, model, device, trainset, model_optimizer, epoch, example_stats):
        model.train()
    
        for batch_idx, batch_start_ind in enumerate(range(0, len(trainset.targets), args.batch_size)):
            batch_inds = npr.permutation(np.arange(len(trainset.targets)))[batch_start_ind:batch_start_ind + args.batch_size]
            inputs = torch.stack([trainset.__getitem__(ind)[0] for ind in batch_inds])
            targets = torch.LongTensor(np.array(trainset.targets)[batch_inds].tolist())
    
            inputs, targets = inputs.to(device), targets.to(device)
            model_optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            model_optimizer.step()
    ```
    
        **설명**
    
    - **모델 학습**: 학습 데이터를 배치로 나누어 모델에 입력하고, 예측값과 실제값의 차이를 계산하여 손실을 구합니다. 그 후, 손실을 역전파하여 모델의 가중치를 업데이트합니다.
    
    **<모델 테스트 (Evaluation Loop)>**
    
    ```python
    def test(epoch, model, device, example_stats):
        model.eval()
    
        with torch.no_grad():
            for batch_idx, batch_start_ind in enumerate(range(0, len(test_dataset.targets), 32)):
                inputs = torch.stack([test_dataset.__getitem__(ind)[0] for ind in range(batch_start_ind, batch_start_ind + 32)])
                targets = torch.LongTensor(np.array(test_dataset.targets)[batch_start_ind:batch_start_ind + 32].tolist())
    
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets).mean()
    ```
    
        **설명**
    
    - **모델 평가**: 테스트 데이터셋을 사용해 학습된 모델의 성능을 평가합니다. 이 과정에서 손실 및 정확도를 계산하여 모델 성능을 확인합니다.
    
    **<통계적 정보 계산 및 저장>**
    
    ```python
    # Forgetting Score를 측정하기 위한 각 예제에 대한 통계적 정보를 저장합니다.
    for j, index in enumerate(batch_inds):
        index_in_original_dataset = train_indx[index]
        margin = output_correct_class.item() - output_highest_incorrect_class.item()
        index_stats = example_stats.get(index_in_original_dataset, [[], [], []])
        index_stats[0].append(loss[j].item())
        index_stats[1].append(acc[j].sum().item())
        index_stats[2].append(margin)
        example_stats[index_in_original_dataset] = index_stats
    
    # 통계적 데이터를 파일로 저장합니다.
    with open(fname + "__stats_dict.pkl", "wb") as f:
        pickle.dump(example_stats, f)
    ```
    
        **설명**
    
    - 각 학습 예제에 대해 손실, 정확도, 마진 정보를 기록합니다.
    
     다음은 order_examples_by_forgetting.py 파일로, 앞서 학습된 모델로부터 얻은 통계적 데이터를 활용하여, 각 데이터 포인트의 Forgetting Score를 계산하는 과정을 다룹니다. Forgetting Score는 학습 중에 특정 데이터 포인트가 모델에 의해 잊혀진 횟수를 기반으로 하여 데이터의 중요도를 평가하는 지표입니다.
    
    **<Forgetting Score 계산>**
    
    ```python
    def compute_forgetting_statistics(diag_stats, npresentations):
        presentations_needed_to_learn = {}
        unlearned_per_presentation = {}
        margins_per_presentation = {}
        first_learned = {}
    
        for example_id, example_stats in diag_stats.items():
            if not isinstance(example_id, str):
                presentation_acc = np.array(example_stats[1][:npresentations])
                transitions = presentation_acc[1:] - presentation_acc[:-1]
    
                if len(np.where(transitions == -1)[0]) > 0:
                    unlearned_per_presentation[example_id] = np.where(
                        transitions == -1)[0] + 2
                else:
                    unlearned_per_presentation[example_id] = []
    
                if len(np.where(presentation_acc == 0)[0]) > 0:
                    presentations_needed_to_learn[example_id] = np.where(
                        presentation_acc == 0)[0][-1] + 1
                else:
                    presentations_needed_to_learn[example_id] = 0
    
                margins_per_presentation = np.array(
                    example_stats[2][:npresentations])
    
                if len(np.where(presentation_acc == 1)[0]) > 0:
                    first_learned[example_id] = np.where(
                        presentation_acc == 1)[0][0]
                else:
                    first_learned[example_id] = np.nan
    
        return presentations_needed_to_learn, unlearned_per_presentation, margins_per_presentation, first_learned
    ```
    
        **설명**
    
    - **기능**: 각 데이터 포인트가 학습 중 얼마나 자주 잊혀졌는지(Forgetting Score), 학습에 필요한 프레젠테이션 수, 첫 번째로 학습된 시점 등을 계산합니다.
    - **활용**: 이 정보를 바탕으로 데이터 포인트를 정렬하고, 중요하지 않은 데이터 포인트를 식별하는 데 사용됩니다.
    
    **<데이터 포인트 정렬 및 저장>**
    
    ```python
    def sort_examples_by_forgetting(unlearned_per_presentation_all,
                                    first_learned_all, npresentations):
        example_original_order = []
        example_stats = []
    
        for example_id in unlearned_per_presentation_all[0].keys():
            example_original_order.append(example_id)
            example_stats.append(0)
    
            for i in range(len(unlearned_per_presentation_all)):
                stats = unlearned_per_presentation_all[i][example_id]
    
                if np.isnan(first_learned_all[i][example_id]):
                    example_stats[-1] += npresentations
                else:
                    example_stats[-1] += len(stats)
    
        return np.array(example_original_order)[np.argsort(example_stats)], np.sort(example_stats)
    ```
    
        **설명**
    
    - **기능**: 각 데이터 포인트의 Forgetting Score를 기준으로 정렬하여, 학습에 덜 중요한 데이터를 선별합니다.
    - **활용**: 정렬된 데이터는 이후의 학습에서 중요하지 않은 데이터를 제거하거나, 학습 데이터의 품질을 개선하는 데 사용됩니다.
    
4. **실험 결과**
    - 세부 과정
        
         먼저, `run_cifar.py` 스크립트를 통해 CIFAR-10 데이터셋에 대한 각 데이터 포인트의 통계적 특성을 학습했습니다. 이 과정에서 100 에포크(epoch) 동안 손실 값, 정확도, 그리고 분류 마진과 같은 통계적 지표들이 수집되었습니다. 이후, `order_examples_by_forgetting.py`를 사용하여 데이터 포인트들을 Forgetting Score 순서대로 정렬했습니다. 이 스코어는 모델이 학습 과정에서 얼마나 쉽게 잊어버리는지를 나타내는 지표입니다.
        
         정렬된 데이터를 바탕으로 다시 `run_cifar.py`를 실행하였으며, 이 과정에서 일부 데이터를 제거하면서 학습을 진행했습니다. 데이터 제거는 두 가지 방식으로 수행되었습니다:
        
        1. **랜덤 제거**: 데이터 포인트를 무작위로 제거.
        2. **선택적 제거**: Forgetting Score가 낮은 데이터(즉, 모델이 잘 잊어버리지 않는 데이터)부터 순차적으로 제거.
        
    - 실험 결과
        
         다음은 제거된 데이터 비율에 따른 모델 성능(정확도)을 비교한 결과입니다.
        
        ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/622a00dc-4c18-4bc1-993f-ce2c31420973/3ad5f592-62b4-465b-a5f8-83adac04f10d/image.png)
        
        - **세로축**: 모델의 정확도
        - **가로축**: 제거된 데이터 비율 (%)
        
         첨부된 그래프에서 파란색 선은 Forgetting Score가 낮은 데이터부터 제거했을 때의 결과를, 주황색 선은 데이터를 무작위로 제거했을 때의 결과를 나타냅니다.
        
         결과를 보면, 제거된 데이터 비율이 증가할수록, 랜덤으로 데이터를 제거했을 때는 모델의 성능이 급격하게 저하되는 것을 알 수 있습니다. 반면, Forgetting Score가 낮은 데이터부터 제거했을 때는 성능이 비교적 안정적으로 유지되었습니다. 이는 모델이 잘 학습하고, 잘 잊지 않는 중요한 데이터를 유지한 상태에서 학습이 이루어졌기 때문으로 해석할 수 있습니다.
        
    - 결론
        
         이 실험을 통해 Forgetting Score를 기반으로 데이터를 정제하는 방법이 모델 성능을 유지하거나, 심지어는 향상시키는 데 도움이 될 수 있음을 확인했습니다. 특히, 중요한 데이터 포인트를 유지한 상태에서 불필요한 데이터를 제거함으로써, 모델의 효율적인 학습이 가능해졌음을 알 수 있습니다.
        
    - 비교분석
        
         추가적으로, 각 클래스별로 Forgetting Score가 높은 이미지와 낮은 이미지의 특성을 분석했습니다. Forgetting Score가 낮은 이미지들은 일반적으로 각 클래스의 특징이 명확하게 드러나며, 분류가 용이한 이미지들이었습니다. 반면, Forgetting Score가 높은 이미지들은 클래스의 일부만을 담거나, 모호한 특성을 가지고 있어 모델이 혼동을 일으킬 가능성이 높은 이미지들이었습니다. 이 차이는 클래스별로 더욱 명확히 드러났습니다
        
        ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/622a00dc-4c18-4bc1-993f-ce2c31420973/42df395d-89ac-4600-a3ac-ca801c5e27ca/image.png)
        
        ---
        
        ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/622a00dc-4c18-4bc1-993f-ce2c31420973/4877638e-260b-4d04-b474-93c0f0e83505/image.png)
        

### 2. 도로 표지판 실험

1. **개요**
    
    도로 표지판 데이터셋을 활용하여 ResNet18 모델을 학습하고, Forgetting Score를 측정하여 데이터셋 내 예제의 학습 영향도를 평가합니다. 이를 통해 데이터셋에서 학습 성능에 부정적인 영향을 미치는 예제를 제거하거나 유지하는 전략을 적용하여 모델 성능을 최적화합니다.
    
2. **실험 과정**
    - **데이터 분석 및 전처리**: Kaggle에서 도로 표지판 데이터를 수집하여 전처리를 수행합니다. 데이터는 크기 조정 및 정규화를 통해 모델 입력에 적합한 형식으로 변환됩니다. 데이터셋은 총 4개의 클래스(cross walk, speed limit, stop, traffic light)로 이루어져 있고, 총 1244개의 이미지로 구성되어 있습니다.
    - **모델 선택 및 학습**: ResNet18 모델을 선택하여 학습을 진행합니다. 학습 과정에서 각 데이터 포인트에 대한 Forgetting Score를 계산하여 중요도를 평가합니다.
    - **데이터셋 최적화**: Forgetting Score가 낮은 데이터 포인트를 제거하여 데이터셋을 최적화한 후, 최적화된 데이터셋으로 모델을 재학습합니다.
    - **모델 평가**: 최종 모델의 성능은 정확도(Accuracy), 혼동 행렬(Confusion Matrix) 등을 통해 평가되며, 최적화 이전과 이후의 성능을 비교합니다.
    
3. **코드 분석**
    
    **<데이터 전처리 및 Augmentation>**
    
    ```python
    # Image Preprocessing
    normalize = transforms.Normalize(
        mean=[0.4640, 0.4723, 0.4972],
        std=[0.1727, 0.1875, 0.1773]
    )
    
    # Setup train transforms
    train_transform = transforms.Compose([
        transforms.Resize(256),  # 이미지를 긴 변 기준으로 256으로 크기 조정
        transforms.RandomResizedCrop(224),  # 무작위로 224x224 크롭
        transforms.RandomHorizontalFlip(),  # 무작위로 좌우 반전
        transforms.ToTensor(),
        normalize,
    ])
    
    # Setup test transforms
    test_transform = transforms.Compose([
        transforms.Resize(256),  # 이미지를 긴 변 기준으로 256으로 크기 조정
        transforms.CenterCrop(224),  # 중앙을 기준으로 224x224 크롭
        transforms.ToTensor(),
        normalize,
    ])
    ```
    
        **설명**
    
    - 이 부분은 이미지 데이터를 전처리하고, 학습 및 테스트 시 사용할 이미지 Augmentation을 정의합니다. 학습 데이터는 다양한 크기와 방향으로 변환되어 모델이 다양한 패턴에 대해 학습할 수 있도록 합니다.
    - 도로 표지판 이미지의 크기를 256으로 조정한 뒤, 무작위로 224x224 크기로 크롭하고, 좌우 반전을 통해 학습 데이터의 다양성을 높입니다. 테스트 데이터는 중앙 크롭만 수행하여 일관된 입력을 제공합니다.
    
    **<데이터 로딩>**
    
    ```python
    data_path = "dataset_2"
    num_classes = 4
    
    # Load the train and test datasets using ImageFolder
    train_dataset = datasets.ImageFolder(
        root=os.path.join(data_path, 'train'),
        transform=train_transform
    )
    
    test_dataset = datasets.ImageFolder(
        root=os.path.join(data_path, 'test'),
        transform=test_transform
    )
    
    # Create DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    ```
    
        **설명**
    
    - 학습 및 테스트 데이터셋을 PyTorch의 `ImageFolder` 클래스를 사용하여 로드하고, 이를 DataLoader를 통해 모델에 전달할 준비를 합니다.
    - `ImageFolder`를 사용하여 이미지와 라벨을 로드하며, DataLoader는 데이터 배치를 처리하고, 멀티 프로세싱을 통해 데이터 로딩 속도를 최적화합니다. 학습 데이터는 셔플되어 모델에 입력되며, 테스트 데이터는 순차적으로 입력됩니다.
    
    이어지는 train, test 및 forgetting score 계산 부분은 앞서 진행된 실험인 cifar10 실험과 일치하여 따로 적지 않았습니다.
    
4. **실험 결과**
    - 실험 과정은 CIFAR10 데이터셋 실험과 유사하게 진행되었습니다. 도로 표지판 데이터셋을 사용하여 각 데이터 포인트의 통계적 특성을 학습하고, Forgetting Score를 계산하였습니다. 그 후, 데이터를 Forgetting Score 순서대로 정렬한 뒤, 일부 데이터를 제거하면서 학습을 진행하였습니다. 총 두 가지 방식으로 실험을 진행하였는데, 하나는 랜덤으로 데이터를 제거한 것이고, 다른 하나는 Forgetting Score가 낮은 데이터부터 제거한 것입니다.
    - 실험 결과는 아래 그림과 같습니다. 세로축은 정확도를, 가로축은 제거된 데이터 비율을 나타냅니다. 파란색 실선은 Forgetting Score가 낮은 데이터부터 제거했을 때의 결과를, 주황색 점선은 랜덤으로 데이터를 제거했을 때의 결과를 나타냅니다. 
     제거된 데이터의 비율이 증가할수록 랜덤으로 데이터를 제거했을 때의 성능은 크게 감소하는 경향을 보였지만, Forgetting Score를 기반으로 데이터를 제거했을 때는 성능이 비교적 안정적으로 유지되었습니다.
        
        ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/622a00dc-4c18-4bc1-993f-ce2c31420973/ae7a3542-9587-43e7-bc82-310bb257f8b3/image.png)
        
    
    - 결과 분석
        
         실험 결과를 통해 각 데이터 포인트의 Forgetting Score에 따라 도로 표지판 이미지가 어떻게 달라지는지 분석했습니다. 학습이 용이했던 이미지를 살펴보면, **Crosswalk** 라벨의 경우 도로 표지판의 주요 특징이 잘 드러나고 배경과의 대비가 높아 쉽게 인식할 수 있었습니다. **Stop** 라벨의 경우에도 선명하고 뚜렷한 'STOP' 표지판 이미지로 빨간색 배경과 하얀색 텍스트가 명확하게 구분되었습니다. **Speed Limit** 라벨의 경우 숫자가 중앙에 위치하고 원형 테두리와 숫자가 명확하게 구분되어 있어 인식이 용이했고, **Traffic Light** 라벨의 경우에도 신호등의 세 가지 색상과 위치가 명확하게 나타나 있으며 각 색상의 대비가 뚜렷했습니다.
        
         반면, 학습이 어려웠던 이미지에서는 특징이 흐릿하거나, 배경과의 대비가 낮아 인식이 어려운 경우가 많았습니다. **Crosswalk** 라벨의 경우 색이 바래져 도로 표지판의 특징이 잘 드러나지 않았고, **Stop** 라벨은 텍스트가 흐릿하고 배경과의 대비가 낮아 인식이 어려웠습니다. **Speed Limit** 라벨은 숫자와 테두리 간의 구분이 명확하지 않았고, **Traffic Light** 라벨은 신호등의 형태가 흐릿해 색상 구분이 어려웠습니다.
        
        ![result.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/622a00dc-4c18-4bc1-993f-ce2c31420973/a4410ee3-79ad-4184-b8b4-ee98c0de7e56/result.png)
        
    
    - CIFAR10 실험과의 비교
        
         CIFAR10 실험에서는 Forgetting Score에 따라 데이터를 제거하는 것이 성능 유지에 있어 뚜렷한 차이를 만들어냈습니다. 특히, CIFAR10 데이터셋은 정제된 이미지들로 구성되어 있어 각 클래스 간의 구분이 명확하고, 데이터의 특성이 잘 보존되었습니다. 이러한 특성으로 인해 Forgetting Score 기반의 데이터 제거가 효과적으로 작용할 수 있었습니다.
        
         반면, 도로 표지판 데이터셋의 경우, 데이터 간의 구분이 모호하고, 이미지 품질이 일정하지 않을 수 있습니다. 이러한 이유로 인해 Forgetting Score를 기반으로 데이터를 제거했을 때도 CIFAR10 데이터셋에서처럼 드라마틱한 차이가 나타나지 않은 것으로 보입니다. 이는 도로 표지판 데이터셋이 CIFAR10에 비해 더 복잡하고 불균일한 특성을 가지고 있기 때문일 수 있습니다.
        
         결과적으로, CIFAR10과 같은 정제된 데이터셋에서는 Forgetting Score를 활용한 데이터 선택이 매우 효과적이었지만, 더 복잡한 실제 환경의 데이터셋에서는 그 효과가 제한적일 수 있음을 시사합니다. 이러한 결과는 데이터셋의 특성과 학습 전략 간의 관계를 이해하는 데 중요한 인사이트를 제공합니다.
