import React from 'react'

export default function ResultsPanel({ result }) {
    if (!result) return null

    const { experiment_type, description, metadata, points } = result

    // Render based on experiment type
    const renderContent = () => {
        switch (experiment_type) {
            case 'wormhole':
                return (
                    <div>
                        <p style={{ color: 'var(--text-secondary)', marginBottom: '1rem' }}>
                            {description}
                        </p>
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                            {points.map((point, i) => (
                                <span
                                    key={i}
                                    style={{
                                        padding: '0.25rem 0.75rem',
                                        background: point.metadata?.is_anchor
                                            ? 'linear-gradient(135deg, rgba(0, 212, 255, 0.2), rgba(255, 0, 170, 0.2))'
                                            : 'var(--bg-tertiary)',
                                        borderRadius: '999px',
                                        fontSize: '0.8rem',
                                        border: point.metadata?.is_anchor
                                            ? '1px solid var(--accent-cyan)'
                                            : '1px solid var(--border-subtle)',
                                    }}
                                >
                                    {i + 1}. {point.label}
                                </span>
                            ))}
                        </div>
                        {metadata?.distance && (
                            <p style={{
                                marginTop: '1rem',
                                fontSize: '0.8rem',
                                color: 'var(--text-muted)',
                                fontFamily: 'var(--font-mono)'
                            }}>
                                Vector distance: {metadata.distance.toFixed(4)}
                            </p>
                        )}
                    </div>
                )

            case 'blackhole':
                const pathColors = {
                    semantic: 'var(--accent-cyan)',
                    emotional: 'var(--accent-magenta)',
                    categorical: 'var(--accent-green)',
                    associative: 'var(--accent-orange)',
                    metaphorical: '#44ff44'
                }
                return (
                    <div>
                        <p style={{ color: 'var(--text-secondary)', marginBottom: '1rem' }}>
                            {description}
                        </p>

                        {/* Start and End */}
                        <div style={{
                            display: 'flex',
                            justifyContent: 'space-between',
                            alignItems: 'center',
                            marginBottom: '1.5rem',
                            padding: '0.75rem',
                            background: 'var(--bg-tertiary)',
                            borderRadius: '8px',
                        }}>
                            <span style={{
                                padding: '0.5rem 1rem',
                                background: 'rgba(0, 212, 255, 0.2)',
                                borderRadius: '999px',
                                border: '1px solid var(--accent-cyan)',
                                fontWeight: 'bold'
                            }}>
                                {metadata?.start}
                            </span>
                            <span style={{ color: 'var(--text-muted)', fontSize: '1.5rem' }}>→</span>
                            <span style={{
                                padding: '0.5rem 1rem',
                                background: 'rgba(255, 0, 170, 0.2)',
                                borderRadius: '999px',
                                border: '1px solid var(--accent-magenta)',
                                fontWeight: 'bold'
                            }}>
                                {metadata?.end}
                            </span>
                        </div>

                        {/* Paths */}
                        {metadata?.paths?.map((path, i) => (
                            <div key={i} style={{
                                marginBottom: '1rem',
                                padding: '0.75rem',
                                background: 'var(--bg-tertiary)',
                                borderRadius: '8px',
                                borderLeft: `3px solid ${pathColors[path.lens] || 'var(--accent-cyan)'}`
                            }}>
                                <h5 style={{
                                    fontSize: '0.75rem',
                                    color: pathColors[path.lens] || 'var(--accent-cyan)',
                                    marginBottom: '0.5rem',
                                    textTransform: 'uppercase',
                                    letterSpacing: '0.1em'
                                }}>
                                    {path.lens} Path
                                </h5>
                                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', alignItems: 'center' }}>
                                    {path.concepts?.map((concept, j) => (
                                        <React.Fragment key={j}>
                                            {j > 0 && <span style={{ color: 'var(--text-muted)' }}>→</span>}
                                            <span style={{
                                                padding: '0.25rem 0.5rem',
                                                background: 'var(--bg-secondary)',
                                                borderRadius: '6px',
                                                fontSize: '0.8rem',
                                                border: `1px solid ${pathColors[path.lens] || 'var(--border-subtle)'}`
                                            }}>
                                                {concept}
                                            </span>
                                        </React.Fragment>
                                    ))}
                                </div>
                            </div>
                        ))}

                        {/* Bridge concepts */}
                        {metadata?.bridges?.length > 0 && (
                            <div style={{
                                marginTop: '1rem',
                                padding: '0.5rem',
                                fontSize: '0.75rem',
                                color: 'var(--text-muted)',
                            }}>
                                <strong>Bridge concepts found:</strong> {metadata.bridges.join(', ')}
                            </div>
                        )}
                    </div>
                )

            case 'supernova':
                return (
                    <div>
                        <p style={{ color: 'var(--text-secondary)', marginBottom: '1rem' }}>
                            {description}
                        </p>
                        <div style={{
                            display: 'grid',
                            gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))',
                            gap: '0.5rem',
                            marginBottom: '1rem'
                        }}>
                            {points.filter(p => p.metadata?.type === 'attribute').map((point, i) => (
                                <span
                                    key={i}
                                    style={{
                                        padding: '0.25rem 0.5rem',
                                        background: 'var(--bg-tertiary)',
                                        borderRadius: '6px',
                                        fontSize: '0.75rem',
                                        border: '1px solid var(--border-subtle)',
                                    }}
                                >
                                    {point.label}
                                </span>
                            ))}
                        </div>
                        {points.find(p => p.metadata?.type === 'anti') && (
                            <div style={{
                                padding: '0.75rem',
                                background: 'rgba(255, 0, 170, 0.1)',
                                borderRadius: '8px',
                                borderLeft: '3px solid var(--accent-magenta)',
                            }}>
                                <span style={{ fontSize: '0.8rem', color: 'var(--accent-magenta)' }}>
                                    Anti-Concept: {points.find(p => p.metadata?.type === 'anti')?.label?.replace('ANTI: ', '')}
                                </span>
                            </div>
                        )}
                    </div>
                )

            case 'mirror':
                return (
                    <div>
                        <p style={{ color: 'var(--text-secondary)', marginBottom: '1rem' }}>
                            {description}
                        </p>
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                            <div>
                                <h5 style={{
                                    fontSize: '0.75rem',
                                    color: 'var(--accent-cyan)',
                                    marginBottom: '0.5rem',
                                    textTransform: 'uppercase',
                                    letterSpacing: '0.1em'
                                }}>Source</h5>
                                {points.filter(p => p.metadata?.type === 'source').map((point, i) => (
                                    <div key={i} style={{
                                        display: 'flex',
                                        alignItems: 'center',
                                        gap: '0.5rem',
                                        marginBottom: '0.25rem'
                                    }}>
                                        <span style={{
                                            color: 'var(--accent-cyan)',
                                            fontSize: '0.75rem'
                                        }}>→</span>
                                        <span style={{ fontSize: '0.85rem' }}>{point.label}</span>
                                    </div>
                                ))}
                            </div>
                            <div>
                                <h5 style={{
                                    fontSize: '0.75rem',
                                    color: 'var(--accent-magenta)',
                                    marginBottom: '0.5rem',
                                    textTransform: 'uppercase',
                                    letterSpacing: '0.1em'
                                }}>Target: {metadata?.target_domain}</h5>
                                {points.filter(p => p.metadata?.type === 'target').map((point, i) => (
                                    <div key={i} style={{
                                        display: 'flex',
                                        alignItems: 'center',
                                        gap: '0.5rem',
                                        marginBottom: '0.25rem'
                                    }}>
                                        <span style={{
                                            color: 'var(--accent-magenta)',
                                            fontSize: '0.75rem'
                                        }}>→</span>
                                        <span style={{ fontSize: '0.85rem' }}>{point.label}</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                )

            case 'steering':
                return (
                    <div>
                        <p style={{ color: 'var(--text-secondary)', marginBottom: '1rem' }}>
                            {description}
                        </p>

                        <div style={{ display: 'grid', gap: '1rem' }}>
                            <div className="result-card original">
                                <h5>Original Output</h5>
                                <p>{metadata?.original_output}</p>
                            </div>

                            <div className="result-card steered">
                                <h5>Steered Output ({metadata?.positive} direction)</h5>
                                <p>{metadata?.steered_output}</p>
                            </div>
                        </div>

                        <div style={{
                            marginTop: '1rem',
                            fontSize: '0.75rem',
                            color: 'var(--text-muted)',
                            fontFamily: 'var(--font-mono)',
                            display: 'flex',
                            gap: '1rem',
                            flexWrap: 'wrap'
                        }}>
                            <span>Layer: {metadata?.layer}</span>
                            <span>Strength: {metadata?.strength}</span>
                            <span>Magnitude: {metadata?.steering_magnitude?.toFixed(4)}</span>
                        </div>
                    </div>
                )

            default:
                return (
                    <div>
                        <p style={{ color: 'var(--text-secondary)' }}>{description}</p>
                        <pre style={{
                            marginTop: '1rem',
                            padding: '1rem',
                            background: 'var(--bg-tertiary)',
                            borderRadius: '8px',
                            fontSize: '0.75rem',
                            overflow: 'auto',
                            maxHeight: '200px'
                        }}>
                            {JSON.stringify(metadata, null, 2)}
                        </pre>
                    </div>
                )
        }
    }

    return (
        <div className="results-panel">
            <h4>
                <span style={{ marginRight: '0.5rem' }}>
                    {experiment_type === 'wormhole' && '🌀'}
                    {experiment_type === 'blackhole' && '🕳️'}
                    {experiment_type === 'supernova' && '💥'}
                    {experiment_type === 'mirror' && '🪞'}
                    {experiment_type === 'steering' && '🧭'}
                    {experiment_type === 'prism' && '💎'}
                </span>
                Results
            </h4>
            {renderContent()}
        </div>
    )
}
