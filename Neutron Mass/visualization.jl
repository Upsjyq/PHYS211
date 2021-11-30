##
using DataFrames, LsqFit, Printf, Plots, DelimitedFiles
pgfplotsx()
using LaTeXStrings

function gaussianBG(y, p)
    return map( x -> (p[1]/(p[3]*sqrt(2*pi))*
        exp(-(x-p[2])^2/(2*p[3]^2)) + p[4]*x + p[5]),
        y)
end

function linearBG(y, p)
        return map( x -> (p[4]*x + p[5]), y)
end

function linear(y, p)
        return map( (x -> p[1] + p[2] *x), y)
end

function readFileCalib(name, skip=22)
        raw = DelimitedFiles.readdlm(joinpath(dataPath, name), '\t', String, '\n', skipstart=skip)
        slice = raw[:, 1:2:3]
        convert = map(x -> parse(Float32, x), slice)
end

ChiSqr(x, y, y_err, model, p) = sum( (y .- model(x,p)) .^ 2 ./ (y_err .^ 2))

dataPath = "C:/Users/jdewh/OneDrive - The University of Chicago/Third Year/PHYS 211/Neutron Mass/Raw data/11_12_21"
plotsPath = "C:/Users/jdewh/OneDrive - The University of Chicago/Third Year/PHYS 211/Neutron Mass/Plot Output"


## Energy-Channel calibration

channels = Vector{Float32}(undef, 5)
dChannels = Vector{Float32}(undef, 5)
energies = Vector{Float32}(undef, 5)
chisqRed = Vector{Float32}(undef, 5)

# fitting calibration Peaks

# Na22 - .511
dpi=200

fileName = "Na22_calibration.tsv"
dataRaw = readFileCalib(fileName)
time = 136.94
lowBound = 160
highBound = 215
xvals = dataRaw[lowBound:highBound, 1]
rates = dataRaw[lowBound:highBound, 2] ./ time
dRates = map(sqrt, dataRaw[lowBound:highBound, 2]) ./ time

Plots.scalefontsizes()
Plots.scalefontsizes(3)

scatter(xvals, rates, yerror=dRates,
        legend=:topright,
        xlabel="Detector channel", 
        ylabel=L"Incidence rate, counts s$^{-1}$", 
        title="Na-22, 511keV emission peak",
        label="Measured rates",
        size=(8*dpi,5*dpi),)
p0 = [100., 190., 10., 0., 0.,]
fit = curve_fit(gaussianBG, xvals, rates, 1 ./ dRates .^ 2, p0)
s = plot!(xvals, gaussianBG(xvals, coef(fit)),color=:red, label="Gaussian fit")
s = plot!(xvals, linearBG(xvals, coef(fit)), color=:gray, label="Linear background",
         show=true)


savefig(s, joinpath(plotsPath, "Na22Calibration.png"))

Plots.scalefontsizes()
Plots.scalefontsizes(2)


channels[1] = coef(fit)[2]
dChannels[1] = stderror(fit)[2]
energies[1] = .511
chisqRed[1] = ChiSqr(xvals, rates, dRates, gaussianBG, coef(fit)) / dof(fit)

scatter(xvals, rates, yerror=dRates, label="Measured rates")
p0 = [100., 190., 10., 0., 0.,]
fit = curve_fit(gaussianBG, xvals, rates, 1 ./ dRates .^ 2, p0)
plot!(xvals, gaussianBG(xvals, coef(fit)),color=:red, label="Gaussian fit")
s1 = plot!(xvals, linearBG(xvals, coef(fit)), color=:gray, 
        label="Linear background", title="Na-22, .511MeV")

# Na22 - 1.27

fileName = "Na22_calibration.tsv"
dataRaw = readFileCalib(fileName)
time = 136.94
lowBound = 420
highBound = 515
xvals = dataRaw[lowBound:highBound, 1]
rates = dataRaw[lowBound:highBound, 2] ./ time
dRates = map(sqrt, dataRaw[lowBound:highBound, 2]) ./ time

scatter(xvals, rates, yerror=dRates, label="Measured rates")
p0 = [100., 460., 10., 0., 0.,]
fit = curve_fit(gaussianBG, xvals, rates, 1 ./ dRates .^ 2, p0)
plot!(xvals, gaussianBG(xvals, coef(fit)),color=:red, label="Gaussian fit")
s2 = plot!(xvals, linearBG(xvals, coef(fit)), color=:gray, 
        label="Linear background", title="Na22, 1.27MeV")


channels[2] = coef(fit)[2]
dChannels[2] = stderror(fit)[2]
energies[2] = 1.27
chisqRed[2] = ChiSqr(xvals, rates, dRates, gaussianBG, coef(fit)) / dof(fit)


# Co-60, 1.17

fileName = "Co60_calibration_good.tsv"
dataRaw = readFileCalib(fileName)
time = 142.23
lowBound = 380
highBound = 460
xvals = dataRaw[lowBound:highBound, 1]
rates = dataRaw[lowBound:highBound, 2] ./ time
dRates = map(sqrt, dataRaw[lowBound:highBound, 2]) ./ time

scatter(xvals, rates, yerror=dRates, label="Measured rates")
p0 = [100., 460., 10., 0., 0.,]
fit = curve_fit(gaussianBG, xvals, rates, 1 ./ dRates .^ 2, p0)
plot!(xvals, gaussianBG(xvals, coef(fit)),color=:red, label="Gaussian fit")
s3 = plot!(xvals, linearBG(xvals, coef(fit)), color=:gray, 
        label="Linear background", title="Co60, 1.17MeV")

channels[3] = coef(fit)[2]
dChannels[3] = stderror(fit)[2]
energies[3] = 1.17
chisqRed[3] = ChiSqr(xvals, rates, dRates, gaussianBG, coef(fit)) / dof(fit)


# Co-60, 1.32

fileName = "Co60_calibration_good.tsv"
dataRaw = readFileCalib(fileName)
time = 142.23
lowBound = 460
highBound = 535
xvals = dataRaw[lowBound:highBound, 1]
rates = dataRaw[lowBound:highBound, 2] ./ time
dRates = map(sqrt, dataRaw[lowBound:highBound, 2]) ./ time

scatter(xvals, rates, yerror=dRates, label="Measured rates")
p0 = [100., 460., 10., 0., 0.,]
fit = curve_fit(gaussianBG, xvals, rates, 1 ./ dRates .^ 2, p0)
plot!(xvals, gaussianBG(xvals, coef(fit)),color=:red, label="Gaussian fit")
s4 = plot!(xvals, linearBG(xvals, coef(fit)), color=:gray, 
        label="Linear background", title="Co60, 1.33MeV")


channels[4] = coef(fit)[2]
dChannels[4] = stderror(fit)[2]
energies[4] = 1.33
chisqRed[4] = ChiSqr(xvals, rates, dRates, gaussianBG, coef(fit)) / dof(fit)


# Cs-137, .662

fileName = "Cs137_calibration.tsv"
dataRaw = readFileCalib(fileName)
time = 201.83
lowBound = 215
highBound = 275
xvals = dataRaw[lowBound:highBound, 1]
rates = dataRaw[lowBound:highBound, 2] ./ time
dRates = map(sqrt, dataRaw[lowBound:highBound, 2]) ./ time

scatter(xvals, rates, yerror=dRates, label="Measured rates")
p0 = [100., 250., 10., 0., 0.,]
fit = curve_fit(gaussianBG, xvals, rates, 1 ./ dRates .^ 2, p0)
plot!(xvals, gaussianBG(xvals, coef(fit)),color=:red, label="Gaussian fit")
s5 = plot!(xvals, linearBG(xvals, coef(fit)), color=:gray, 
        label="Linear background", title="Cs-137, .662MeV")


channels[5] = coef(fit)[2]
dChannels[5] = stderror(fit)[2]
energies[5] = .662
chisqRed[5] = ChiSqr(xvals, rates, dRates, gaussianBG, coef(fit)) / dof(fit)

## stacked plots

stack = plot(s1, s2, s3, s4, s5, size = (dpi* 8, dpi * 10), 
        layout = (5,1), legend =:topright, xlabel="Detector channel", 
        ylabel=L"rate, count s$^{-1}$",
        ms=2, msw=.7,
        show=true)
gui(stack)
savefig(stack, joinpath(plotsPath, "stackedSpectrum.png"))



## linear regression for channels-energies
p0 = [0., 340.]
fit = curve_fit(linear, energies, channels, 1 ./ dChannels .^ 2, p0)
calibEnergy = 1/ coef(fit)[2]
calibEnergyErr = stderror(fit)[2] / coef(fit)[2] ^ 2
##

scatter(energies, channels, yerror = dChannels, 
        label="Measured Peaks", title = "Energy calibration",
        xlabel="Energy, MeV", ylabel="Detector channel",
        legend=:topleft, 
        ms = 2, msw = 1
        )

calibPlot = plot!(energies, linear(energies, coef(fit)),
        label="linear fit", color=:red,
        show=true)

##

println("channels/MeV: %f +- %f \n", (coef(fit)[2], stderror(fit)[2]))
println("y-intercept: %f +- %f \n", (coef(fit)[1], stderror(fit)[1]))

println("reduced chi squared: ", ChiSqr(energies, channels, dChannels, linear, coef(fit)) / dof(fit))

println("energy / channel: ", 1 / coef(fit)[2])
println("percent error in channel / energy: ", stderror(fit)[2] / coef(fit)[2] * 100.)
println("error in energy / channel: ", stderror(fit)[2] / (coef(fit)[2] ^ 2) )

## Plotting the CH800 Peaks

fileName = "beamopen_pb100mm+fullofparaffin.tsv"
dataRaw = DelimitedFiles.readdlm(joinpath(dataPath, fileName), '\t', Float32, '\n', skipstart=27) 
xvals = dataRaw[700:900, 1]

t1 = 260.15 #from tsv file, line 17
counts1 = dataRaw[700:900, 3]
dCounts1 = map(sqrt, counts1)
rates1 = counts1 ./ t1
dRates1 = dCounts1 ./ t1


fileName = "beamopen_pb100mm.tsv"
dataRaw = DelimitedFiles.readdlm(joinpath(dataPath, fileName), '\t', Float32, '\n', skipstart=27) 
xvals = dataRaw[700:900, 1]

t2 = 265.0 #from tsv file, line 17
counts2 = dataRaw[700:900, 3]
dCounts2 = map(sqrt, counts2)
rates2 = counts2 ./ t2
dRates2 = dCounts2 ./ t2


fileName = "beamopen_pb100mm+fullofgraphite.tsv"
dataRaw = DelimitedFiles.readdlm(joinpath(dataPath, fileName), '\t', Float32, '\n', skipstart=27)
xvals = dataRaw[700:900, 1]

t3 = 211.06 #from tsv file, line 17
counts3 = dataRaw[700:900,3]
dCounts3 = map(sqrt, counts3)
rates3 = counts3 ./ t3
dRates3 = dCounts3 ./ t3


##
amps = Vector{Float32}(undef, 3)
ampErrs = Vector{Float32}(undef, 3)
locs = Vector{Float32}(undef, 3)
locErrs = Vector{Float32}(undef, 3)
redchisq = Vector{Float32}(undef, 3)

p01 = [12., 800., 20., 0., .8]
wts = 1 ./ dRates1 .^ 2
fit1 = curve_fit(gaussianBG, xvals, rates1, wts, p01)
amps[1] = coef(fit1)[1]
locs[1] = coef(fit1)[2]
ampErrs[1] = stderror(fit1)[1]
locErrs[1] = stderror(fit1)[2]
redchisq[1] = ChiSqr(xvals, rates1, dRates1, gaussianBG, coef(fit1)) / dof(fit1)

p02 = [4., 800., 20., 0., .8]
wts = 1 ./ dRates2 .^ 2
fit2 = curve_fit(gaussianBG, xvals, rates2, wts, p02)
amps[2] = coef(fit2)[1]
locs[2] = coef(fit2)[2]
ampErrs[2] = stderror(fit2)[1]
locErrs[2] = stderror(fit2)[2]
redchisq[2] = ChiSqr(xvals, rates2, dRates2, gaussianBG, coef(fit2)) / dof(fit2)

p03 = [.2, 810., 19., 0., .6]
wts = 1 ./ dRates3 .^ 2
fit3 = curve_fit(gaussianBG, xvals, rates3, wts, p03)
amps[3] = coef(fit3)[1]
locs[3] = coef(fit3)[2]
ampErrs[3] = stderror(fit3)[1]
locErrs[3] = stderror(fit3)[2]
redchisq[3] = ChiSqr(xvals, rates3, dRates3, gaussianBG, coef(fit3)) /dof(fit3)

println("peak energy of peak 1: ", locs[1] * calibEnergy)
println("peak energy error: ", locs[1] * calibEnergy * sqrt( (locErrs[1]/locs[1])^2 + (calibEnergyErr/calibEnergy)^2))


##

Plots.scalefontsizes()
Plots.scalefontsizes(2)

dpi=200
s1 = scatter(xvals, rates1, yerror = dRates1, 
        ms=2, msw=.7, label="Measured Rates", 
        title="2.3 MeV peak, 100mm Lead Shielding + Paraffin Blocks", 
        ylabel = "Detection Rate, counts /s",
        left_margin=15Plots.mm, right_margin=5Plots.mm,
        top_margin=5Plots.mm, bottom_margin=10Plots.mm,
        legend=false)
s1 = plot!(xvals, linearBG(xvals, coef(fit1)), 
        lw=3, color=:gray, label="Fit, Linear Background")
s1 = plot!(xvals, gaussianBG(xvals, coef(fit1)), 
        lw=3, color=:red, label="Gaussian Fit")

s2 = scatter(xvals, rates2, yerror = dRates2, 
        ms=2, msw=.7, label="Measured Rates", 
        title="2.3 MeV peak, 100mm Lead Shielding", 
        ylabel = "Detection Rate, counts /s", 
        left_margin=15Plots.mm, right_margin=5Plots.mm,
        top_margin=5Plots.mm, bottom_margin=10Plots.mm,
        legend=false)
s2 = plot!(xvals, linearBG(xvals, coef(fit2)), 
        lw=3, color=:gray, label="Fit, Linear Background")
s2 = plot!(xvals, gaussianBG(xvals, coef(fit2)), 
        lw=3, color=:red, label="Gaussian Fit")

s3 = scatter(xvals, rates3, yerror = dRates3, 
        ms=2, msw=.7, label="Measured Rates", 
        title="2.3 MeV peak, 100mm Lead Shielding + Graphite Blocks", 
        xlabel = "Detector Channel", 
        ylabel = "Detection Rate, counts /s", 
        left_margin=15Plots.mm, right_margin=5Plots.mm,
        top_margin=5Plots.mm, bottom_margin=10Plots.mm,
        legend=false)
s3 = plot!(xvals, linearBG(xvals, coef(fit3)), 
        lw=3, color=:gray, label="Fit, Linear Background")
s3 = plot!(xvals, gaussianBG(xvals, coef(fit3)), 
        lw=3, color=:red, label="Gaussian Fit")


s = plot(s1, s2, s3, size=(10*dpi, 8*dpi), layout=(3,1),
        legend=:topright, show=true)
gui(s)
savefig(s, joinpath(plotsPath, "beamOpenComparison.png"))

