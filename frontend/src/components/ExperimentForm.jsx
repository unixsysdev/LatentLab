import React, { useState } from 'react'

// Default values for each experiment
const DEFAULT_VALUES = {
    wormhole: {
        start: 'Medieval Castle',
        end: 'Space Station',
        steps: 7
    },
    supernova: {
        concept: 'Apple',
        num_attributes: 15
    },
    mirror: {
        source_chain: ['Rome Rise', 'Rome Peak', 'Rome Fall'],
        target_domain: 'A Dubstep Track'
    },
    steering: {
        prompt: 'I feel really angry about',
        positive_concept: 'Love',
        negative_concept: 'Hate',
        layer: 15,
        strength: 1.5
    }
}

export default function ExperimentForm({ experimentId, onSubmit, isLoading }) {
    const defaults = DEFAULT_VALUES[experimentId] || {}
    const [values, setValues] = useState(defaults)

    // Reset values when experiment changes
    React.useEffect(() => {
        setValues(DEFAULT_VALUES[experimentId] || {})
    }, [experimentId])

    const handleChange = (field, value) => {
        setValues(prev => ({ ...prev, [field]: value }))
    }

    const handleSubmit = (e) => {
        e.preventDefault()
        onSubmit(values)
    }

    // Render form based on experiment type
    const renderFields = () => {
        switch (experimentId) {
            case 'wormhole':
                return (
                    <>
                        <div className="form-group">
                            <label>Start Concept</label>
                            <input
                                type="text"
                                className="form-input"
                                value={values.start || ''}
                                onChange={(e) => handleChange('start', e.target.value)}
                                placeholder="e.g., Medieval Castle"
                            />
                        </div>
                        <div className="form-group">
                            <label>End Concept</label>
                            <input
                                type="text"
                                className="form-input"
                                value={values.end || ''}
                                onChange={(e) => handleChange('end', e.target.value)}
                                placeholder="e.g., Space Station"
                            />
                        </div>
                        <div className="form-group">
                            <label>Steps</label>
                            <div className="slider-container">
                                <input
                                    type="range"
                                    min="3"
                                    max="15"
                                    value={values.steps || 7}
                                    onChange={(e) => handleChange('steps', parseInt(e.target.value))}
                                />
                                <span className="slider-value">{values.steps || 7}</span>
                            </div>
                        </div>
                    </>
                )

            case 'supernova':
                return (
                    <>
                        <div className="form-group">
                            <label>Concept to Explode</label>
                            <input
                                type="text"
                                className="form-input"
                                value={values.concept || ''}
                                onChange={(e) => handleChange('concept', e.target.value)}
                                placeholder="e.g., Apple, Love, Technology"
                            />
                        </div>
                        <div className="form-group">
                            <label>Number of Dimensions</label>
                            <div className="slider-container">
                                <input
                                    type="range"
                                    min="5"
                                    max="25"
                                    value={values.num_attributes || 15}
                                    onChange={(e) => handleChange('num_attributes', parseInt(e.target.value))}
                                />
                                <span className="slider-value">{values.num_attributes || 15}</span>
                            </div>
                        </div>
                    </>
                )

            case 'mirror':
                return (
                    <>
                        <div className="form-group">
                            <label>Source Chain (one per line)</label>
                            <textarea
                                className="form-input"
                                value={(values.source_chain || []).join('\n')}
                                onChange={(e) => handleChange('source_chain',
                                    e.target.value.split('\n').filter(s => s.trim())
                                )}
                                placeholder="Rome Rise&#10;Rome Peak&#10;Rome Fall"
                                rows={4}
                            />
                        </div>
                        <div className="form-group">
                            <label>Target Domain</label>
                            <input
                                type="text"
                                className="form-input"
                                value={values.target_domain || ''}
                                onChange={(e) => handleChange('target_domain', e.target.value)}
                                placeholder="e.g., A Dubstep Track"
                            />
                        </div>
                    </>
                )

            case 'steering':
                return (
                    <>
                        <div className="form-group">
                            <label>Prompt</label>
                            <textarea
                                className="form-input"
                                value={values.prompt || ''}
                                onChange={(e) => handleChange('prompt', e.target.value)}
                                placeholder="e.g., I feel really angry about"
                                rows={2}
                            />
                        </div>
                        <div className="form-group">
                            <label>Steer Towards (+)</label>
                            <input
                                type="text"
                                className="form-input"
                                value={values.positive_concept || ''}
                                onChange={(e) => handleChange('positive_concept', e.target.value)}
                                placeholder="e.g., Love, Joy, Peace"
                            />
                        </div>
                        <div className="form-group">
                            <label>Steer Away From (-)</label>
                            <input
                                type="text"
                                className="form-input"
                                value={values.negative_concept || ''}
                                onChange={(e) => handleChange('negative_concept', e.target.value)}
                                placeholder="e.g., Hate, Anger, Fear"
                            />
                        </div>
                        <div className="form-group">
                            <label>Injection Layer</label>
                            <div className="slider-container">
                                <input
                                    type="range"
                                    min="1"
                                    max="30"
                                    value={values.layer || 15}
                                    onChange={(e) => handleChange('layer', parseInt(e.target.value))}
                                />
                                <span className="slider-value">{values.layer || 15}</span>
                            </div>
                        </div>
                        <div className="form-group">
                            <label>Strength</label>
                            <div className="slider-container">
                                <input
                                    type="range"
                                    min="0.1"
                                    max="5"
                                    step="0.1"
                                    value={values.strength || 1.5}
                                    onChange={(e) => handleChange('strength', parseFloat(e.target.value))}
                                />
                                <span className="slider-value">{(values.strength || 1.5).toFixed(1)}</span>
                            </div>
                        </div>
                    </>
                )

            default:
                return <p>Unknown experiment type</p>
        }
    }

    return (
        <form onSubmit={handleSubmit}>
            {renderFields()}

            <button
                type="submit"
                className="btn btn-primary btn-full"
                disabled={isLoading}
            >
                {isLoading ? (
                    <>
                        <span className="loading-spinner" style={{ width: 16, height: 16 }}></span>
                        Running...
                    </>
                ) : (
                    <>
                        <span>🚀</span>
                        Run Experiment
                    </>
                )}
            </button>
        </form>
    )
}
