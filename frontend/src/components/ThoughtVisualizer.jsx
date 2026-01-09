import React, { useRef, useMemo, useState, useEffect } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import {
    OrbitControls,
    Line,
    Text,
    Sphere,
    Box,
    Html,
    PerspectiveCamera
} from '@react-three/drei'
import * as THREE from 'three'

// Colors matching our CSS theme
const COLORS = {
    primary: '#00d4ff',
    secondary: '#ff00aa',
    tertiary: '#00ff88',
    accent: '#ffaa00',
    grid: '#1a1a2e',
    text: '#e0e0e0',
    positive: '#44ff44',
    negative: '#ff4444',
}

// Animated point component
function AnimatedPoint({ position, color, size = 0.15, label, metadata = {} }) {
    const meshRef = useRef()
    const [hovered, setHovered] = useState(false)

    useFrame((state) => {
        if (meshRef.current) {
            meshRef.current.scale.setScalar(hovered ? 1.5 : 1)
        }
    })

    return (
        <group position={position}>
            <mesh
                ref={meshRef}
                onPointerOver={() => setHovered(true)}
                onPointerOut={() => setHovered(false)}
            >
                <sphereGeometry args={[size, 32, 32]} />
                <meshStandardMaterial
                    color={color}
                    emissive={color}
                    emissiveIntensity={hovered ? 0.8 : 0.3}
                />
            </mesh>

            {/* Label */}
            {label && (
                <Html
                    position={[0, size + 0.15, 0]}
                    center
                    distanceFactor={10}
                    style={{
                        transition: 'all 0.2s',
                        opacity: hovered ? 1 : 0.7,
                        transform: `scale(${hovered ? 1.1 : 1})`,
                    }}
                >
                    <div className="point-label" style={{
                        background: hovered ? 'rgba(0, 212, 255, 0.2)' : 'rgba(10, 10, 18, 0.8)',
                        borderColor: hovered ? COLORS.primary : 'rgba(255,255,255,0.1)',
                    }}>
                        {label}
                    </div>
                </Html>
            )}
        </group>
    )
}

// Glowing line between points
function GlowingLine({ points, color = COLORS.primary, lineWidth = 2 }) {
    return (
        <Line
            points={points}
            color={color}
            lineWidth={lineWidth}
            transparent
            opacity={0.8}
        />
    )
}

// Animated trajectory (for wormhole)
function TrajectoryPath({ points, colors }) {
    const lineRef = useRef()

    const curvePoints = useMemo(() => {
        if (points.length < 2) return []
        const vectors = points.map(p => new THREE.Vector3(...p))
        const curve = new THREE.CatmullRomCurve3(vectors)
        return curve.getPoints(50)
    }, [points])

    return (
        <>
            {curvePoints.length > 1 && (
                <Line
                    points={curvePoints}
                    color={COLORS.primary}
                    lineWidth={3}
                    transparent
                    opacity={0.8}
                />
            )}
        </>
    )
}

// Starburst pattern for Supernova
function Starburst({ center, rays, antiPoint }) {
    return (
        <group>
            {rays.map((ray, i) => (
                <GlowingLine
                    key={i}
                    points={[center, ray.coords]}
                    color={COLORS.primary}
                    lineWidth={1}
                />
            ))}
        </group>
    )
}

// Grid floor
function GridFloor() {
    return (
        <gridHelper
            args={[20, 40, COLORS.grid, COLORS.grid]}
            position={[0, -2, 0]}
            rotation={[0, 0, 0]}
        />
    )
}

// Scene with environment
function Scene({ experimentResult, experimentType }) {
    const { camera } = useThree()

    // Normalize coordinates to fit in a viewable range
    const normalizedPoints = useMemo(() => {
        if (!experimentResult?.points?.length) return null

        const coords = experimentResult.points.map(p => p.coords_3d)

        // Find bounds
        let minX = Infinity, maxX = -Infinity
        let minY = Infinity, maxY = -Infinity
        let minZ = Infinity, maxZ = -Infinity

        coords.forEach(([x, y, z]) => {
            minX = Math.min(minX, x); maxX = Math.max(maxX, x)
            minY = Math.min(minY, y); maxY = Math.max(maxY, y)
            minZ = Math.min(minZ, z); maxZ = Math.max(maxZ, z)
        })

        // Calculate scale to fit in a ~10 unit box
        const rangeX = maxX - minX || 1
        const rangeY = maxY - minY || 1
        const rangeZ = maxZ - minZ || 1
        const maxRange = Math.max(rangeX, rangeY, rangeZ)
        const scale = 8 / maxRange

        // Center point
        const centerX = (minX + maxX) / 2
        const centerY = (minY + maxY) / 2
        const centerZ = (minZ + maxZ) / 2

        // Normalize each point
        return experimentResult.points.map(p => ({
            ...p,
            coords_3d: [
                (p.coords_3d[0] - centerX) * scale,
                (p.coords_3d[1] - centerY) * scale,
                (p.coords_3d[2] - centerZ) * scale
            ]
        }))
    }, [experimentResult])

    // Auto-fit camera to normalized points
    useEffect(() => {
        if (normalizedPoints?.length > 0) {
            camera.position.set(8, 5, 8)
            camera.lookAt(0, 0, 0)
        }
    }, [normalizedPoints, camera])

    if (!normalizedPoints) {
        return (
            <>
                <GridFloor />
                {/* Placeholder content */}
                <Text
                    position={[0, 0, 0]}
                    fontSize={0.3}
                    color={COLORS.grid}
                    anchorX="center"
                    anchorY="middle"
                >
                    Run an experiment to visualize
                </Text>
            </>
        )
    }

    const points = normalizedPoints
    const connections = experimentResult.connections || []

    // Color mapping based on point metadata
    const getPointColor = (point, index) => {
        const type = point.metadata?.type
        if (type === 'center') return COLORS.accent
        if (type === 'anti') return COLORS.secondary
        if (type === 'source') return COLORS.primary
        if (type === 'target') return COLORS.secondary
        if (type === 'positive') return COLORS.positive
        if (type === 'negative') return COLORS.negative
        if (type === 'original') return COLORS.primary
        if (type === 'steered') return COLORS.secondary
        if (type === 'prompt') return COLORS.text

        // Gradient for wormhole
        if (experimentType === 'wormhole') {
            const t = index / Math.max(1, points.length - 1)
            return new THREE.Color(COLORS.secondary).lerp(
                new THREE.Color(COLORS.tertiary), t
            ).getStyle()
        }

        return COLORS.primary
    }

    const getPointSize = (point, index) => {
        const type = point.metadata?.type
        if (type === 'center') return 0.25
        if (type === 'anti') return 0.2
        if (point.metadata?.is_anchor) return 0.2
        return 0.12
    }

    return (
        <>
            <GridFloor />

            {/* Draw connections */}
            {connections.map(([i, j], idx) => {
                if (i >= points.length || j >= points.length) return null
                const p1 = points[i].coords_3d
                const p2 = points[j].coords_3d

                // Different styling for mirror connections
                const isMirrorConnection =
                    points[i].metadata?.type !== points[j].metadata?.type &&
                    (points[i].metadata?.type === 'source' || points[i].metadata?.type === 'target')

                return (
                    <GlowingLine
                        key={idx}
                        points={[p1, p2]}
                        color={isMirrorConnection ? COLORS.grid : COLORS.primary}
                        lineWidth={isMirrorConnection ? 1 : 2}
                    />
                )
            })}

            {/* Draw trajectory for wormhole */}
            {experimentType === 'wormhole' && points.length > 2 && (
                <TrajectoryPath
                    points={points.map(p => p.coords_3d)}
                />
            )}

            {/* Draw points */}
            {points.map((point, i) => (
                <AnimatedPoint
                    key={i}
                    position={point.coords_3d}
                    color={getPointColor(point, i)}
                    size={getPointSize(point, i)}
                    label={point.label}
                    metadata={point.metadata}
                />
            ))}
        </>
    )
}

export default function ThoughtVisualizer({ experimentResult, experimentType }) {
    return (
        <Canvas
            camera={{ position: [5, 3, 5], fov: 60 }}
            gl={{ antialias: true, alpha: true }}
            style={{ background: 'transparent' }}
        >
            <color attach="background" args={['#050508']} />

            {/* Lighting */}
            <ambientLight intensity={0.4} />
            <pointLight position={[10, 10, 10]} intensity={0.6} color="#ffffff" />
            <pointLight position={[-10, 5, -10]} intensity={0.3} color={COLORS.primary} />
            <pointLight position={[0, -5, 5]} intensity={0.2} color={COLORS.secondary} />

            {/* Scene */}
            <Scene
                experimentResult={experimentResult}
                experimentType={experimentType}
            />

            {/* Controls */}
            <OrbitControls
                enableDamping
                dampingFactor={0.05}
                rotateSpeed={0.5}
                zoomSpeed={0.8}
                minDistance={2}
                maxDistance={50}
            />
        </Canvas>
    )
}
