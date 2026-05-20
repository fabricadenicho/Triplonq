# Indicador — MNQ×CL Divergência

**Arquivo:** `indicador - div_cl.pine`
**Tipo:** Pane separado (overlay=false)
**Ativos:** MNQ1! + CL1! (comparativo)

## O que faz

Plota em pane separado a divergência de preço normalizada entre MNQ e CL barra a barra. É a versão "pura" do sinal de divergência que alimenta os outros indicadores — útil para analisar a intensidade e duração da divergência em detalhe.

## Como funciona

Calcula `price_div_cl = (retorno_MNQ - retorno_CL) × scale` onde scale padrão = 10.000. O histograma verde significa MNQ e CL movendo juntos (positivo = MNQ liderando), vermelho significa divergência (CL e MNQ se afastando).

`strong_div` é marcado quando ADX >= limiar AND divergência significativa — esse é o mesmo sinal que entra nos indicadores A++ de CL e MNQ.

## Indicadores no pane

- **Histograma colorido** — verde = juntos/positivo, vermelho = divergindo
- **Linha zero** — referência
- **Marcador strong_div** — ponto laranja quando ADX + divergência atingem limiar

## Configurações

- `ADX Length`: 17
- `ADX Threshold`: 14
- `Scale ×`: 10.000 (para facilitar leitura visual)
- `Símbolos`: configuráveis (padrão MNQ1! e CL1!)
- `Só sessão US (09-17h UTC)`: desativado por padrão

## Quando usar

Abrir em pane separado abaixo do gráfico principal do MNQ ou CL para monitorar a divergência em tempo real. Quando o histograma vira vermelho forte + ADX sobe = sinal A++ se aproximando.
