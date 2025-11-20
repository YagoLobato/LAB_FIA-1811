# LTNtorch — Classificação Binária com Lógica Tensorial (2D e Cães vs. Gatos)

Este projeto demonstra o uso do **LTNtorch** para aprender modelos neurais guiados por
conhecimento lógico (Logic Tensor Networks, LTN) em dois cenários:

1. Um exemplo **2D sintético** de classificação binária (predicado `A(x)`), já presente no notebook original.
2. Um exemplo **real de classificação Cão vs. Gato**, inspirado diretamente no modelo do artigo:

> T. Carraro, L. Serafini, F. Aiolli. **LTNtorch: PyTorch Implementation of Logic Tensor Networks**.  
> Journal of Machine Learning Research, 2024. (preprint em https://arxiv.org/abs/2409.16045)

O notebook base utilizado é o `LTNTorch_Fia.ipynb`, agora estendido com uma seção adicional
para classificação de cães e gatos usando imagens e uma CNN.

---

## 1. Visão Geral: LTN e o Artigo

O LTNtorch implementa o framework **Logic Tensor Networks (LTN)** em PyTorch. A ideia central é:

- Representar **predicados, funções, constantes e variáveis lógicas** como tensores ("grounding");  
- Fazer com que predicados retornem valores em $[0, 1]$, interpretados como valores de verdade fuzzy;
- Implementar conectivos lógicos ($\land, \lor, \lnot, \Rightarrow$) com operadores de lógica fuzzy;
- Tratar quantificadores ($\forall, \exists$) como **agregadores diferenciáveis** sobre conjuntos de exemplos;
- Definir a função de perda a partir da **satisfação lógica** de um conjunto de fórmulas $K = \{\varphi_i\}$.

Dado um conjunto de fórmulas fechadas $K$, a perda típica é:

$$\mathcal{L}(\theta) = 1 - \text{SatAgg}_{\varphi \in K} G(\varphi \mid \theta),$$

onde:

- $G(\varphi \mid \theta)$ é o valor de verdade (fuzzy) da fórmula $\varphi$ dado o modelo parametrizado por $\theta$;
- `SatAgg` agrega as satisfações de todas as fórmulas em uma medida única.

O artigo do LTNtorch apresenta, entre outros exemplos, um caso de **classificação binária Cão vs. Gato**
em que um predicado `Dog(x)` é implementado como uma CNN, e a lógica expressa que:

- imagens de cães devem satisfazer `Dog(x)`;
- imagens de gatos devem satisfazer `¬Dog(x)`.

---

## 2. Estrutura do Notebook

O notebook `LTNTorch_Fia.ipynb` (e a variante `LTNTorch_Fia_catsdogs.ipynb`) está organizado em duas partes principais:

### 2.1. Parte 1 — Exemplo 2D Sintético (Predicado `A(x)`)

Esta é a parte original do notebook, que implementa um exemplo simples de LTN:

1. **Geração de dados 2D**  
   - `data_pos`: pontos próximos ao centro $(0.5, 0.5)$ → classe positiva;  
   - `data_neg`: pontos fora do centro → classe negativa.

2. **Modelo MLP e Predicado `A(x)`**
   - `MLP(2, hidden, 1)` recebe vetores 2D e produz um escalar em $[0,1]$;
   - `A_model = MLP(...)` é embrulhado em `A = ltntorch.Predicate(A_model)`.

3. **Lógica LTN**
   - variáveis: `x_A = Variable("x_A", data_pos)`, `x_B = Variable("x_B", data_neg)`;
   - axiomas:
     - $\forall x_A\; A(x_A)$;
     - $\forall x_B\; \lnot A(x_B)$;
   - agregador: `sat = SatAgg(axiom_A, axiom_B)`;
   - perda: `loss = 1.0 - sat`.

4. **Treinamento e Visualização**
   - loop de épocas com `loss.backward()` e `optimizer.step()`;
   - visualização da fronteira de decisão em 2D (heatmap de `A(x)` e pontos de treino).

> Essa parte serve como uma introdução conceitual: como mapear um modelo PyTorch para um predicado LTN,
> definir axiomas e treinar via satisfação lógica.

---

### 2.2. Parte 2 — Classificação Cão vs. Gato (Modelo do Artigo)

Na nova seção **"Código: 6. Classificação Binária Cão vs. Gato (Modelo do Artigo)"**, o notebook estende
o mesmo raciocínio para um problema real de visão computacional.

A ideia é replicar a estrutura do exemplo 2D, mas com:

- Base de imagens com duas classes (Dog / Cat);
- CNN como predicado `Dog(x)`;
- Axiomas $\forall dog\; Dog(dog)$ e $\forall cat\; \lnot Dog(cat)$.

#### 2.2.1. Download e Preparação do Dataset (KaggleHub)

Usamos o pacote **`kagglehub`** para baixar o dataset:

> `bhavikjikadara/dog-and-cat-classification-dataset`

O fluxo da célula é:

1. Instalar `kagglehub` (no Colab, por exemplo):
   ```bash
   pip install kagglehub
   ```

2. Baixar o dataset e localizar a pasta `PetImages` com `Cat` e `Dog`:

   ```python
   import kagglehub
   from pathlib import Path
   import shutil

   DATA_ROOT = Path("data/cats_dogs")
   DOG_DIR = DATA_ROOT / "dogs"
   CAT_DIR = DATA_ROOT / "cats"

   DOG_DIR.mkdir(parents=True, exist_ok=True)
   CAT_DIR.mkdir(parents=True, exist_ok=True)

   path = kagglehub.dataset_download("bhavikjikadara/dog-and-cat-classification-dataset")
   print("Path retornado pelo kagglehub:", path)

   petimages = Path(path) / "PetImages"
   dog_src = petimages / "Dog"
   cat_src = petimages / "Cat"

   # Copia Dog → data/cats_dogs/dogs
   # Copia Cat → data/cats_dogs/cats
   ```

3. Após essa etapa, a estrutura final utilizada no notebook é:

   ```text
   data/
     cats_dogs/
       dogs/   # imagens de cães
       cats/   # imagens de gatos
   ```

#### 2.2.2. Dataset e DataLoaders

Definimos uma classe simples de dataset baseada em diretórios:

```python
class SimpleImageFolder(Dataset):
    def __init__(self, root_dir: Path, transform=None):
        self.root_dir = Path(root_dir)
        exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        self.paths = []
        for ext in exts:
            self.paths.extend(sorted(self.root_dir.glob(ext)))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img
```

E os `DataLoader`s:

```python
dog_dataset = SimpleImageFolder(DOG_DIR, transform=transform)
cat_dataset = SimpleImageFolder(CAT_DIR, transform=transform)

dog_loader = DataLoader(dog_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
cat_loader = DataLoader(cat_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
```

Esses datasets fazem o papel de `data_pos` (cães) e `data_neg` (gatos) do exemplo 2D.

#### 2.2.3. CNN e Predicado `Dog(x)`

Implementamos uma CNN simples `CNN_Dog` como modelo base para o predicado:

```python
class CNN_Dog(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

cnn_dog_model = CNN_Dog().to(ltntorch.device)
Dog = ltntorch.Predicate(cnn_dog_model)
```

E, análogo ao exemplo 2D, definimos:

```python
Not_img    = ltntorch.Connective(ltntorch.fuzzy_ops.NotStandard())
Forall_img = ltntorch.Quantifier(ltntorch.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
SatAgg_img = ltntorch.fuzzy_ops.SatAgg()

optimizer_img = torch.optim.Adam(Dog.parameters(), lr=1e-4)
```

#### 2.2.4. Loop de Treinamento (Dog vs. Cat)

O loop é o paralelo direto do 2D:

```python
for epoch in range(n_epochs):
    epoch_loss = 0.0

    for dog_imgs, cat_imgs in zip(dog_loader, cat_loader):
        optimizer_img.zero_grad()

        dog_imgs = dog_imgs.to(ltntorch.device)
        cat_imgs = cat_imgs.to(ltntorch.device)

        dog = ltntorch.Variable("dog", dog_imgs)
        cat = ltntorch.Variable("cat", cat_imgs)

        phi1 = Forall_img(dog, Dog(dog))           # ∀dog Dog(dog)
        phi2 = Forall_img(cat, Not_img(Dog(cat)))  # ∀cat ¬Dog(cat)

        sat_agg = SatAgg_img(phi1, phi2)
        loss = 1.0 - sat_agg

        loss.backward()
        optimizer_img.step()

        epoch_loss += loss.item()

    epoch_loss /= max(1, len(dog_loader))
    print(f"[Dog vs Cat] Epoch {epoch+1}/{n_epochs} - Sat: {sat_agg.item():.4f} - Loss: {epoch_loss:.4f}")
```

- No lugar de `x_A`/`x_B`, usamos `dog`/`cat`;
- No lugar de `A(x)`, usamos `Dog(x)` (CNN);
- A função de perda é a mesma ideia: `1 - SatAgg` sobre os axiomas.

#### 2.2.5. Visualização de Exemplos

Para inspecionar qualitativamente o que o predicado `Dog(x)` aprendeu, o notebook mostra
algumas imagens de cães e gatos com os respectivos valores de verdade:

```python
def show_examples(loader, title):
    imgs = next(iter(loader))
    imgs = imgs.to(ltntorch.device)

    # Importante: embrulhar o tensor em uma Variable LTN
    x = ltntorch.Variable("x", imgs)

    with torch.no_grad():
        preds_obj = Dog(x)        # LTNObject
        preds = preds_obj.value   # tensor com valores em [0, 1]
        preds = preds.cpu().numpy().flatten()

    imgs = imgs.cpu()
    n = min(8, imgs.size(0))
    plt.figure(figsize=(16, 4))
    plt.suptitle(title)
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(imgs[i].permute(1, 2, 0))
        plt.axis("off")
        plt.title(f"Dog(x)={preds[i]:.2f}")
    plt.show()

show_examples(dog_loader, "Exemplos de CÃES com valores Dog(x)")
show_examples(cat_loader, "Exemplos de GATOS com valores Dog(x)")
```

Note que, para usar o predicado LTN corretamente, precisamos passar um `LTNObject` (Variable)
para `Dog`, de forma semelhante ao que é feito no loop de treino.

---

## 3. Como Executar (Google Colab)

1. **Suba o notebook (`LTNTorch_Fia.ipynb` ou `LTNTorch_Fia_catsdogs.ipynb`) no Colab.**

2. **Instale as dependências principais** na primeira célula (ou logo após):

```bash
pip install torch torchvision pillow LTNtorch kagglehub
```

3. **Execute a célula de download/preparo da base Cão vs. Gato**  
   (a que usa `kagglehub.dataset_download("bhavikjikadara/dog-and-cat-classification-dataset")`).

4. **Execute as células da Parte 1 (2D)** se quiser revisar o exemplo sintético.

5. **Execute as células da Parte 2 (Cão vs. Gato)** na ordem:
   - dataset + DataLoaders,
   - CNN + predicado `Dog(x)`,
   - loop de treinamento,
   - visualização de exemplos.

---

## 4. Referências

- T. Carraro, L. Serafini, F. Aiolli. **LTNtorch: PyTorch Implementation of Logic Tensor Networks**, JMLR, 2024.
- S. Badreddine, A. d'Avila Garcez, L. Serafini, M. Spranger. **Logic Tensor Networks**, Artificial Intelligence, 303, 2022.
- Repositório oficial LTNtorch: https://github.com/tommasocarraro/LTNtorch
- Documentação LTNtorch: https://tommasocarraro.github.io/LTNtorch/