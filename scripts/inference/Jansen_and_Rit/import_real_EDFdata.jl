# EDF file: European Data Format
# Data downloaded from https://archive.physionet.org/pn6/chbmit/
# Each file contains 1 hour of data with a frequency observation of 1/250 seconds
# • in between two files there are at most 10 seconds gaps
# • in total there are several hours of data (massive dataframe)
# • data are int16
# • there are several channel, right now we choose one and we model that one
using EDF
using DataFrames
using CSV
using Plots

OUT_DIR = joinpath(Base.source_dir(), "..", "output")
INPUT = joinpath(Base.source_dir(), "..", "data")
patient = "chb01_01"


file_name = joinpath(INPUT, String(patient)*".edf")
data =  EDF.read(file_name)

seconds = 100
channel = 1
Time = 1/256:1/256:seconds
x1 = Array{Float64,1}(data.signals[channel].samples[1:length(Time)])

plot(Time, x1, title = string(seconds)*" seconds sample, patient "*String(patient), label = "Channel "*string(channel) )
FIG_PATH = "./assets/realdata.pdf"
savefig(FIG_PATH)


#Export dataframe
FILENAME_OUT = joinpath(OUT_DIR,
                        "real_data_"*string(seconds)*"_seconds.csv")

df = DataFrame(time=Time, x1 = x1, x2 = [NaN for i in 1:length(Time)])
CSV.write(FILENAME_OUT, df)
