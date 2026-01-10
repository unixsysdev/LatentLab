import React, { useState, useEffect, useCallback } from 'react'
import ThoughtVisualizer from './components/ThoughtVisualizer'
import ExperimentForm from './components/ExperimentForm'
import ResultsPanel from './components/ResultsPanel'

// Experiment configurations
const EXPERIMENTS = [
    {
        id: 'wormhole',
        name: 'Wormhole',
        icon: '🌀',
        description: 'Visualize the semantic trajectory between two distant concepts. Watch how meaning morphs through the latent space.',
    },
    {
        id: 'blackhole',
        name: 'Blackhole',
        icon: '🕳️',
        description: 'Find multiple semantic paths between two concepts using activation analysis and LLM-guided discovery. See how different semantic lenses (emotional, categorical, associative) create different routes.',
    },
    {
        id: 'supernova',
        name: 'Supernova',
        icon: '💥',
        description: 'Explode a concept into its semantic dimensions. See the high-dimensional features that make up any idea.',
    },
    {
        id: 'prism',
        name: 'Concept Prism',
        icon: '💎',
        description: 'Spectrally analyze a concept using reverse embedding. See the exact semantic components that make up a concept vector.',
    },
    {
        id: 'mirror',
        name: 'Mirror',
        icon: '🪞',
        description: 'Map relationship structures across domains. Watch how "Rome Rise → Peak → Fall" maps onto completely different topics.',
    },
    {
        id: 'steering',
        name: 'Steering',
        icon: '🧭',
        description: 'Inject activation vectors to alter model behavior. The most "magical" experiment - manipulate thoughts directly.',
    },
]

function App() {
    const [selectedExperiment, setSelectedExperiment] = useState('wormhole')
    const [experimentResult, setExperimentResult] = useState(null)
    const [isLoading, setIsLoading] = useState(false)
    const [error, setError] = useState(null)
    const [serverStatus, setServerStatus] = useState('loading')
    const [models, setModels] = useState([])
    const [currentModel, setCurrentModel] = useState(null)
    const [isModelSwitching, setIsModelSwitching] = useState(false)

    // Fetch available models
    useEffect(() => {
        const fetchModels = async () => {
            try {
                const res = await fetch('/api/models')
                if (res.ok) {
                    const data = await res.json()
                    setModels(data.models || [])
                    setCurrentModel(data.current)
                }
            } catch (e) {
                console.error('Failed to fetch models:', e)
            }
        }
        fetchModels()
    }, [])

    // Check server health on mount
    useEffect(() => {
        const checkHealth = async () => {
            try {
                const res = await fetch('/api/health')
                if (res.ok) {
                    const data = await res.json()
                    setServerStatus(data.status)
                } else {
                    setServerStatus('error')
                }
            } catch (e) {
                setServerStatus('error')
            }
        }

        checkHealth()
        const interval = setInterval(checkHealth, 10000)
        return () => clearInterval(interval)
    }, [])

    const handleModelSwitch = async (modelId) => {
        if (modelId === currentModel || isModelSwitching) return

        setIsModelSwitching(true)
        setError(null)

        try {
            const res = await fetch('/api/models/switch', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model_id: modelId })
            })

            if (res.ok) {
                setCurrentModel(modelId)
                setExperimentResult(null) // Clear old result
            } else {
                const data = await res.json()
                setError(data.detail || 'Failed to switch model')
            }
        } catch (e) {
            setError('Failed to switch model: ' + e.message)
        } finally {
            setIsModelSwitching(false)
        }
    }

    const currentExperiment = EXPERIMENTS.find(e => e.id === selectedExperiment)

    const handleRunExperiment = useCallback(async (inputs) => {
        setIsLoading(true)
        setError(null)

        try {
            const res = await fetch(`/api/experiment/${selectedExperiment}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(inputs)
            })

            if (!res.ok) {
                const errData = await res.json()
                throw new Error(errData.detail || 'Experiment failed')
            }

            const result = await res.json()
            setExperimentResult(result)
        } catch (e) {
            setError(e.message)
            console.error('Experiment error:', e)
        } finally {
            setIsLoading(false)
        }
    }, [selectedExperiment])

    return (
        <div className="app">
            {/* Header */}
            <header className="header">
                <div className="logo">
                    <div className="logo-icon">🧠</div>
                    <h1>LatentLab</h1>
                </div>

                {/* Model Selector */}
                <div className="model-selector">
                    <label>Model:</label>
                    <select
                        value={currentModel || ''}
                        onChange={(e) => handleModelSwitch(e.target.value)}
                        disabled={isModelSwitching}
                    >
                        {models.map(m => (
                            <option key={m.id} value={m.id}>
                                {m.name} ({m.size})
                            </option>
                        ))}
                    </select>
                    {isModelSwitching && <span className="model-loading">Loading...</span>}
                </div>

                <div className="status-badge">
                    <div className={`status-dot ${serverStatus}`}></div>
                    <span>
                        {serverStatus === 'ok' ? 'Connected' :
                            serverStatus === 'loading' ? 'Connecting...' : 'Offline'}
                    </span>
                </div>
            </header>

            <div className="main-container">
                {/* Sidebar */}
                <aside className="sidebar">
                    <div className="sidebar-section">
                        <h3>Experiments</h3>
                        <div className="experiment-grid">
                            {EXPERIMENTS.map(exp => (
                                <button
                                    key={exp.id}
                                    className={`experiment-card ${selectedExperiment === exp.id ? 'active' : ''}`}
                                    onClick={() => {
                                        setSelectedExperiment(exp.id)
                                        setExperimentResult(null)
                                        setError(null)
                                    }}
                                    title={exp.description}
                                >
                                    <span className="experiment-card-icon">{exp.icon}</span>
                                    <span className="experiment-card-name">{exp.name}</span>
                                </button>
                            ))}
                        </div>
                    </div>

                    {currentExperiment && (
                        <div className="sidebar-section">
                            <div className="experiment-description">
                                <p>{currentExperiment.description}</p>
                            </div>

                            <ExperimentForm
                                experimentId={currentExperiment.id}
                                onSubmit={handleRunExperiment}
                                isLoading={isLoading}
                            />

                            {error && (
                                <div style={{
                                    color: '#ff4444',
                                    marginTop: '1rem',
                                    padding: '0.5rem',
                                    background: 'rgba(255, 68, 68, 0.1)',
                                    borderRadius: '6px',
                                    fontSize: '0.85rem'
                                }}>
                                    {error}
                                </div>
                            )}
                        </div>
                    )}
                </aside>

                {/* Main Visualization Area */}
                <main className="main-content">
                    <div className="canvas-container">
                        {isLoading && (
                            <div className="loading-overlay">
                                <div style={{ textAlign: 'center' }}>
                                    <div className="loading-spinner"></div>
                                    <p className="loading-text">Running experiment...</p>
                                </div>
                            </div>
                        )}

                        <ThoughtVisualizer
                            experimentResult={experimentResult}
                            experimentType={selectedExperiment}
                        />
                    </div>

                    {experimentResult && (
                        <ResultsPanel result={experimentResult} />
                    )}
                </main>
            </div>
        </div >
    )
}

export default App
