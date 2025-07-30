import React, { useState } from 'react';

const App = () => {
    const [ticker, setTicker] = useState('b3sa3.SA');
    const [period, setPeriod] = useState('1y'); 
    const [loading, setLoading] = useState(false);
    const [lstmResults, setLstmResults] = useState(null);
    const [biasVarianceResults, setBiasVarianceResults] = useState(null);
    const [error, setError] = useState(null);

    // IMPORTANTE: Substitua este URL pelo URL do seu backend Python implantado no Render!
    // Exemplo: 'https://seu-nome-do-backend.onrender.com/analyze_stock'
    const backendApiUrl = 'https://previsao-5-dias-bias-lstm.onrender.com'; 

    const runAnalysis = async () => {
        setLoading(true);
        setError(null);
        setLstmResults(null);
        setBiasVarianceResults(null);

        try {
            const response = await fetch(backendApiUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ ticker: ticker, period: period })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `Erro HTTP: ${response.status}`);
            }

            const parsedResults = await response.json();
            
            if (parsedResults.error) {
                setError(parsedResults.error);
            } else {
                setLstmResults(parsedResults.lstm_results);
                setBiasVarianceResults(parsedResults.bias_variance_results);
            }

        } catch (fetchError) {
            setError(`Erro na comunicação com o backend: ${fetchError.message}. Certifique-se de que o backend Python está rodando e acessível.`);
        } finally {
            setLoading(false);
        }
    };

    const formatPercentage = (value) => {
        if (typeof value === 'number' && !isNaN(value)) {
            return `${(value * 100).toFixed(2)}%`;
        }
        return 'N/A';
    };

    const formatCurrency = (value) => {
        if (typeof value === 'number' && !isNaN(value)) {
            return `R$ ${value.toFixed(2)}`;
        }
        return 'N/A';
    };

    return (
        <div className="min-h-screen bg-gray-100 p-4 font-sans flex flex-col items-center">
            <div className="bg-white p-6 rounded-lg shadow-md w-full max-w-4xl mb-6">
                <h1 className="text-3xl font-bold text-center text-gray-800 mb-6">Análise Unificada de Ações</h1>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                    <div>
                        <label htmlFor="ticker" className="block text-sm font-medium text-gray-700 mb-1">
                            Ticker da Ação (ex: b3sa3.SA, ^BVSP)
                        </label>
                        <input
                            type="text"
                            id="ticker"
                            className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                            value={ticker}
                            onChange={(e) => setTicker(e.target.value)}
                            placeholder="Ex: b3sa3.SA"
                        />
                    </div>
                    <div>
                        <label htmlFor="period" className="block text-sm font-medium text-gray-700 mb-1">
                            Período (ex: 1y, 5y, 10y)
                        </label>
                        <input
                            type="text"
                            id="period"
                            className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                            value={period}
                            onChange={(e) => setPeriod(e.target.value)}
                            placeholder="Ex: 5y"
                        />
                    </div>
                </div>

                <button
                    onClick={runAnalysis}
                    disabled={loading}
                    className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                    {loading ? 'Executando Análise...' : 'Executar Análise Completa'}
                </button>
            </div>

            {error && (
                <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative w-full max-w-4xl mb-6" role="alert">
                    <strong className="font-bold">Erro:</strong>
                    <span className="block sm:inline ml-2">{error}</span>
                </div>
            )}

            {loading && (
                <div className="flex items-center justify-center w-full max-w-4xl mb-6 text-blue-600 text-lg">
                    <svg className="animate-spin -ml-1 mr-3 h-8 w-8 text-blue-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Processando dados e modelos... Isso pode levar alguns minutos.
                </div>
            )}

            {lstmResults && biasVarianceResults && (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 w-full max-w-6xl">
                    {/* Seção de Previsão LSTM */}
                    <div className="bg-white p-6 rounded-lg shadow-md">
                        <h2 className="text-2xl font-semibold text-gray-800 mb-4 border-b pb-2">Resultados da Previsão LSTM</h2>
                        {lstmResults.error ? (
                            <p className="text-red-600">{lstmResults.error}</p>
                        ) : (
                            <>
                                <p className="mb-2"><strong className="text-gray-700">Ticker:</strong> {lstmResults.ticker}</p>
                                <p className="mb-2"><strong className="text-gray-700">Preço Atual (último do histórico):</strong> {formatCurrency(lstmResults.current_price)}</p>

                                <h3 className="text-xl font-medium text-gray-700 mt-4 mb-2">Métricas do Modelo (conjunto de teste):</h3>
                                <ul className="list-disc list-inside ml-4 mb-4">
                                    <li><strong className="text-gray-600">RMSE:</strong> {lstmResults.metrics?.rmse?.toFixed(4) || 'N/A'}</li>
                                    <li><strong className="text-gray-600">MAE:</strong> {lstmResults.metrics?.mae?.toFixed(4) || 'N/A'}</li>
                                    <li><strong className="text-gray-600">R²:</strong> {lstmResults.metrics?.r2?.toFixed(4) || 'N/A'}</li>
                                </ul>

                                <h3 className="text-xl font-medium text-gray-700 mt-4 mb-2">Previsões para os Próximos {lstmResults.predictions?.length || 0} Dias:</h3>
                                <div className="space-y-2">
                                    {lstmResults.predictions && lstmResults.predictions.map((pred, index) => {
                                        const change = pred - lstmResults.current_price;
                                        const change_pct = (change / lstmResults.current_price) * 100;
                                        return (
                                            <p key={index} className="text-gray-800">
                                                <strong className="text-gray-600">{lstmResults.dates[index]}:</strong> {formatCurrency(pred)} ({change > 0 ? '+' : ''}{change.toFixed(2)}, {change_pct > 0 ? '+' : ''}{change_pct.toFixed(1)}%)
                                            </p>
                                        );
                                    })}
                                </div>
                                <p className="text-sm text-gray-500 mt-4">
                                    {/* Gráficos de previsão e indicadores (RSI, MACD) seriam exibidos aqui em uma implementação completa. */}
                                </p>
                            </>
                        )}
                    </div>

                    {/* Seção de Análise de Bias-Variância */}
                    <div className="bg-white p-6 rounded-lg shadow-md">
                        <h2 className="text-2xl font-semibold text-gray-800 mb-4 border-b pb-2">Resultados da Análise de Bias-Variância</h2>
                        {biasVarianceResults.error ? (
                            <p className="text-red-600">{biasVarianceResults.error}</p>
                        ) : (
                            <>
                                <h3 className="text-xl font-medium text-gray-700 mt-4 mb-2">Decomposição Bias-Variância:</h3>
                                {biasVarianceResults.bias_variance_decomposition && typeof biasVarianceResults.bias_variance_decomposition === 'object' ? (
                                    <div className="overflow-x-auto mb-4">
                                        <table className="min-w-full divide-y divide-gray-200">
                                            <thead className="bg-gray-50">
                                                <tr>
                                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Modelo</th>
                                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Erro Total</th>
                                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Bias</th>
                                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Variância</th>
                                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Erro Irredutível</th>
                                                </tr>
                                            </thead>
                                            <tbody className="bg-white divide-y divide-gray-200">
                                                {Object.entries(biasVarianceResults.bias_variance_decomposition).map(([modelName, metrics]) => (
                                                    <tr key={modelName}>
                                                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{modelName}</td>
                                                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{metrics['Total Error']?.toFixed(4) || 'N/A'}</td>
                                                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{metrics['Bias']?.toFixed(4) || 'N/A'}</td>
                                                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{metrics['Variance']?.toFixed(4) || 'N/A'}</td>
                                                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{metrics['Irreducible Error']?.toFixed(4) || 'N/A'}</td>
                                                    </tr>
                                                ))}
                                            </tbody>
                                        </table>
                                    </div>
                                ) : (
                                    <p className="text-gray-600">{biasVarianceResults.bias_variance_decomposition}</p>
                                )}

                                <h3 className="text-xl font-medium text-gray-700 mt-4 mb-2">Métricas da Estratégia de Trading:</h3>
                                {biasVarianceResults.strategy_metrics && typeof biasVarianceResults.strategy_metrics === 'object' ? (
                                    <ul className="list-disc list-inside ml-4 mb-4">
                                        <li><strong className="text-gray-600">Capital Final:</strong> {formatCurrency(biasVarianceResults.strategy_metrics['Capital Final'])}</li>
                                        <li><strong className="text-gray-600">Retorno Total:</strong> {formatPercentage(biasVarianceResults.strategy_metrics['Retorno Total'])}</li>
                                        <li><strong className="text-gray-600">CAGR:</strong> {formatPercentage(biasVarianceResults.strategy_metrics['CAGR'])}</li>
                                        <li><strong className="text-gray-600">Sharpe Ratio:</strong> {biasVarianceResults.strategy_metrics['Sharpe Ratio']?.toFixed(2) || 'N/A'}</li>
                                        <li><strong className="text-gray-600">Max Drawdown:</strong> {formatPercentage(biasVarianceResults.strategy_metrics['Max Drawdown'])}</li>
                                        <li><strong className="text-gray-600">Hit Ratio:</strong> {formatPercentage(biasVarianceResults.strategy_metrics['Hit Ratio'])}</li>
                                        <li><strong className="text-gray-600">Retorno Médio por Trade:</strong> {formatPercentage(biasVarianceResults.strategy_metrics['Retorno Médio por Trade'])}</li>
                                    </ul>
                                ) : (
                                    <p className="text-gray-600">{biasVarianceResults.strategy_metrics}</p>
                                )}

                                <h3 className="text-xl font-medium text-gray-700 mt-4 mb-2">Métricas do Buy and Hold:</h3>
                                {biasVarianceResults.buy_and_hold_metrics && typeof biasVarianceResults.buy_and_hold_metrics === 'object' ? (
                                    <ul className="list-disc list-inside ml-4 mb-4">
                                        <li><strong className="text-gray-600">Capital Final:</strong> {formatCurrency(biasVarianceResults.buy_and_hold_metrics['Capital Final'])}</li>
                                        <li><strong className="text-gray-600">Retorno Total:</strong> {formatPercentage(biasVarianceResults.buy_and_hold_metrics['Retorno Total'])}</li>
                                        <li><strong className="text-gray-600">CAGR:</strong> {formatPercentage(biasVarianceResults.buy_and_hold_metrics['CAGR'])}</li>
                                        <li><strong className="text-gray-600">Sharpe Ratio:</strong> {biasVarianceResults.buy_and_hold_metrics['Sharpe Ratio']?.toFixed(2) || 'N/A'}</li>
                                        <li><strong className="text-gray-600">Max Drawdown:</strong> {formatPercentage(biasVarianceResults.buy_and_hold_metrics['Max Drawdown'])}</li>
                                    </ul>
                                ) : (
                                    <p className="text-gray-600">{biasVarianceResults.buy_and_hold_metrics}</p>
                                )}

                                <h3 className="text-xl font-medium text-gray-700 mt-4 mb-2">Previsão Futura (Bias-Variância):</h3>
                                {biasVarianceResults.future_prediction && typeof biasVarianceResults.future_prediction === 'object' ? (
                                    <p className="text-gray-800">
                                        <strong className="text-gray-600">Última Data Disponível:</strong> {biasVarianceResults.future_prediction.date_available}<br />
                                        <strong className="text-gray-600">Previsão de Retorno (5 dias):</strong> {formatPercentage(biasVarianceResults.future_prediction.prediction_value)}<br />
                                        <strong className="text-gray-600">Direção:</strong> {biasVarianceResults.future_prediction.direction}
                                    </p>
                                ) : (
                                    <p className="text-gray-600">{biasVarianceResults.future_prediction}</p>
                                )}
                                <p className="text-sm text-gray-500 mt-4">
                                    {/* Gráficos de previsão e indicadores (RSI, MACD) seriam exibidos aqui em uma implementação completa. */}
                                </p>
                            </>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
};

export default App;
