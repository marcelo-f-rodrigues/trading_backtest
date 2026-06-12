"""Estratégias e documentação amigável para o dashboard."""

STRATEGY_DOCUMENTATION = {
    "MovingAverageCrossover": {
        "friendly_name": "Cruze de Médias Móveis",
        "summary": "Compra quando a média curta fica acima da média longa e fica em cash quando o cruzamento vira contra a posição.",
        "indicators": ["Média móvel rápida", "Média móvel lenta"],
        "entry_trigger": "Preço fecha acima da média curta e esta acima da média longa.",
        "exit_trigger": "A média curta cruza abaixo da média longa ou a tendência perde força.",
        "parameters": "fast, slow, ma_type",
    },
    "MovingAverageFilter": {
        "friendly_name": "Filtro de Tendência",
        "summary": "Opera apenas quando o preço está acima de uma média móvel de referência, reduzindo ruído em mercados laterais.",
        "indicators": ["Preço de fechamento", "Média móvel"],
        "entry_trigger": "Fechamento acima da média móvel escolhida.",
        "exit_trigger": "Fechamento abaixo da média móvel escolhida.",
        "parameters": "period, ma_type",
    },
    "TripleMovingAverage": {
        "friendly_name": "Três Médias Móveis",
        "summary": "Compra quando a estrutura de médias móveis está alinhada com a tendência de alta, evitando entradas em cenários confusos.",
        "indicators": ["Média curta", "Média intermediária", "Média longa"],
        "entry_trigger": "short > mid > long.",
        "exit_trigger": "Qualquer quebra da sequência de alinhamento.",
        "parameters": "short, mid, long, ma_type",
    },
    "DonchianBreakout": {
        "friendly_name": "Breakout de Donchian",
        "summary": "Entra quando o preço rompe a máxima recente e sai quando cai para a mínima recente do canal de saída.",
        "indicators": ["Máxima dos últimos N dias", "Mínima dos últimos M dias"],
        "entry_trigger": "Fechamento acima da máxima recente do canal de entrada.",
        "exit_trigger": "Fechamento abaixo da mínima recente do canal de saída.",
        "parameters": "entry_period, exit_period",
    },
    "PriceReturnMomentum": {
        "friendly_name": "Momentum de Retorno",
        "summary": "Busca manter exposição quando o ativo apresenta desempenho recente forte em relação ao passado.",
        "indicators": ["Retorno acumulado", "Lookback de período"],
        "entry_trigger": "Retorno recente supera o limiar configurado.",
        "exit_trigger": "Retorno recente fica abaixo do limiar ou perde força.",
        "parameters": "lookback, threshold",
    },
    "RateOfChange": {
        "friendly_name": "Taxa de Variação",
        "summary": "Identifica aceleração ou desaceleração do preço em uma janela recente para filtrar entradas fortes.",
        "indicators": ["ROC", "Período de referência"],
        "entry_trigger": "ROC acima de zero ou acima de um limiar.",
        "exit_trigger": "ROC cai abaixo do limiar definido.",
        "parameters": "period, threshold",
    },
    "RSIReversion": {
        "friendly_name": "Reversão via RSI",
        "summary": "Compra em sobrevenda e tenta sair em sobrecompra, explorando reversão após extremos de preço.",
        "indicators": ["RSI"],
        "entry_trigger": "RSI cai abaixo do nível de sobrevenda.",
        "exit_trigger": "RSI sobe acima do nível de sobrecompra.",
        "parameters": "period, oversold, overbought",
    },
    "RSIBands": {
        "friendly_name": "Bandas de RSI",
        "summary": "Versão mais conservadora do RSI, usando um nível de entrada menor e saída neutra para filtrar ruído.",
        "indicators": ["RSI", "Nível de compra", "Nível de saída"],
        "entry_trigger": "RSI abaixo do nível de compra.",
        "exit_trigger": "RSI sobe acima do nível de saída.",
        "parameters": "period, buy_level, exit_level",
    },
    "BollingerReversion": {
        "friendly_name": "Reversão com Bandas de Bollinger",
        "summary": "Entra quando o preço se afasta da média e sai ao voltar ao centro da banda ou ao nível definido.",
        "indicators": ["Média móvel", "Desvio padrão", "Bandas de Bollinger"],
        "entry_trigger": "Preço ultrapassa a banda inferior ou o nível de z-score configurado.",
        "exit_trigger": "Preço volta à média ou à banda superior.",
        "parameters": "period, std_dev, exit_at",
    },
    "ZScoreReversion": {
        "friendly_name": "Reversão por Z-Score",
        "summary": "Usa z-score do preço para medir desvio em relação à média histórica e operar reversões extremas.",
        "indicators": ["Z-score", "Média móvel"],
        "entry_trigger": "Z-score ultrapassa o limite de entrada.",
        "exit_trigger": "Z-score retorna ao nível neutro definido.",
        "parameters": "period, entry_z, exit_z",
    },
    "TrendFilteredReversion": {
        "friendly_name": "Reversão com Filtro de Tendência",
        "summary": "Só opera reversão quando o preço está em contexto de tendência, reduzindo sinais ruins em mercados laterais.",
        "indicators": ["Média de tendência", "RSI"],
        "entry_trigger": "Preço acima da média longa e RSI em sobrevenda.",
        "exit_trigger": "RSI sobe demais ou a tendência perde força.",
        "parameters": "trend_period, rsi_period, oversold, overbought",
    },
    "MomentumReversion": {
        "friendly_name": "Momentum + Reversão",
        "summary": "Compra pullbacks em contexto de momentum positivo, aproveitando correções dentro de tendências fortes.",
        "indicators": ["Momentum de médio prazo", "Retorno de curto prazo"],
        "entry_trigger": "Momentum positivo e pullback recente bastante forte.",
        "exit_trigger": "Curto prazo retorna a território positivo.",
        "parameters": "momentum_period, reversion_period, momentum_threshold, reversion_threshold",
    },
    "DCA_Weekly": {
        "friendly_name": "DCA Semanal",
        "summary": "Benchmark de compra programada semanal, investindo valor fixo em intervalos regulares ao longo do tempo.",
        "indicators": ["Preço de fechamento", "Valor fixo por semana"],
        "entry_trigger": "Agendamento semanal fixo.",
        "exit_trigger": "Benchmark de comparação, sem saída ativa.",
        "parameters": "initial_capital, contribution, frequency",
    },
    "DCA_Monthly": {
        "friendly_name": "DCA Mensal",
        "summary": "Benchmark de compra programada mensal, útil para comparar com estratégias mais ativas.",
        "indicators": ["Preço de fechamento", "Valor fixo por mês"],
        "entry_trigger": "Agendamento mensal fixo.",
        "exit_trigger": "Benchmark de comparação, sem saída ativa.",
        "parameters": "initial_capital, contribution, frequency",
    },
}


def get_strategy_documentation(strategy_name: str) -> dict:
    """Retorna documento amigável para a estratégia, com fallback seguro."""
    base = STRATEGY_DOCUMENTATION.get(strategy_name, {})
    return {
        "friendly_name": base.get("friendly_name", strategy_name),
        "summary": base.get("summary", "Estratégia do framework de backtest."),
        "indicators": base.get("indicators", []),
        "entry_trigger": base.get("entry_trigger", "Ver lógica da estratégia específica."),
        "exit_trigger": base.get("exit_trigger", "Ver lógica da estratégia específica."),
        "parameters": base.get("parameters", "Parâmetros configuráveis"),
    }
