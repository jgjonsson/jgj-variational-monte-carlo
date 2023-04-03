# Problem1h implementation

Here we compare radial distributions for repulsive hamiltonian and harmonic oscillator hamiltonian.

# Results recreation

Recreating this takes some presetup, but is possible. I assume one runs the code from the directory with this README.

## Running sampler

One first needs to change the sampler. Within `sample` function, add following
```cpp
    auto dist = [](Particle p)
    {
        auto pos = p.getPosition();
        return sqrt(pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]);
    };
    for (auto &p : system->getParticles())
    {
        cout << dist(*p) << '\n';
    }
```
and recompile (not elegant, but we do not aim for it). Then build and run `sample_distance.cpp` with various hard core radii (for example 0.0043 and 0.0000043), and save the output to files.

## Plotting

One has to have ROOT installed to continue. Build and run `density_superimpose.cpp` with `root-config --cflags --glibs` as compiler flag.