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
        id: 'supernova',
        name: 'Supernova',
        icon: '💥',
        description: 'Explode a concept into its semantic dimensions. See the high-dimensional features that make up any idea.',
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
                <div className="status-badge">
                    <div className={`status-dot ${serverStatus}`}></div>
                    <span>
                        {serverStatus === 'ok' ? 'Connected' :
                            serverStatus === 'loading' ? 'Connecting...' : 'Offline'}
                    </span>
                </div>
            </header>

            {/* Sidebar */}
            <aside className="sidebar">
                <div className="sidebar-section">
                    <h3>Experiments</h3>
                    <div className="experiment-tabs">
                        {EXPERIMENTS.map(exp => (
                            <button
                                key={exp.id}
                                className={`experiment-tab ${selectedExperiment === exp.id ? 'active' : ''}`}
                                onClick={() => {
                                    setSelectedExperiment(exp.id)
                                    setExperimentResult(null)
                                    setError(null)
                                }}
                            >
                                {exp.icon} {exp.name}
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
    )
}

export default App
