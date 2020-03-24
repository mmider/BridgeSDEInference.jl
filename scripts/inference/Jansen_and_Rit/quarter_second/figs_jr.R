setwd("~/Sync/DOCUMENTS/onderzoek/Presentaties/2020/neuro_10mintalk/data_jr_sebastiano")
library(tidyverse)
library(hablar)

theme_set(theme_bw())

######## Read data 
obs <- read_csv("df1_new.csv")
paths <- read_csv("df2.csv") %>% 
        pivot_longer(cols=contains("x"), names_to='component') %>% mutate(iter=iter*100) %>%
  mutate(component=factor(component,labels=c("x[1]", "x[2]", "x[3]", "x[4]", "x[5]", "x[6]")))

params <- read_csv("df3.csv") %>% 
  pivot_longer(cols=c(par_a, par_b, par_C, par_muy, par_sigmay), names_to='parameter') %>%
   mutate(parameter=factor(parameter, labels=c("a","b","C","mu[y]", "sigma[y]")))

trueparams <- data.frame(parameter=unique(params$parameter), vals=c(100,50,135,220,2000))%>%
  mutate(parameter=factor(parameter, labels=c("a","b","C","mu[y]", "sigma[y]")))
  
truepath <- read_csv("df4.csv") %>% pivot_longer(cols=contains("x"), names_to='component') %>%
  mutate(component=factor(component,labels=c("x[1]", "x[2]", "x[3]", "x[4]", "x[5]", "x[6]")))

longpath <- read_csv("df5.csv")

######## making figs
pathssub <- paths %>% filter(iter %% 1000 ==0 )
paths_fig <- pathssub %>%   
    ggplot() +
    geom_path(aes(x=time,y=value,colour=iter,group=iter),size=0.4)  +
    facet_wrap(~component,label="label_parsed",scales='free') +
  scale_colour_gradient(low = "orange", high = "grey") +
  ylab("")

paths_fig_withtruth <- pathssub %>%   
  ggplot() +
  geom_path(aes(x=time,y=value,colour=iter,group=iter),size=0.4)  +
  geom_path(data=truepath, aes(x=time, y=value)) +
  facet_wrap(~component,label="label_parsed",scales='free') +
  scale_colour_gradient(low = "orange", high = "grey") +
  ylab("")


params_fig <-
  params %>% filter(iter %% 10 ==0) %>%
  ggplot(aes(x=iter,y=value,group=parameter)) +
  geom_path(size=0.4) + 
  geom_hline(data=trueparams, aes(yintercept=vals),colour='yellow')+
  facet_wrap(~parameter,label="label_parsed",scales='free') + xlab("iteration")+ylab("")

params_fig2 <-
  params %>% filter(iter>10000) %>%
  ggplot(aes(x=value,group=parameter)) +
  geom_histogram() + 
  geom_vline(data=trueparams, aes(xintercept=vals),colour='yellow')+
  facet_wrap(~parameter,label="label_parsed",scales='free') + xlab("iteration")+ylab("")


pathsdiff <- read_csv("df2.csv") %>%   mutate(iter=iter*100, diff =x2-x3) %>%
 select(-contains("x")) %>% filter(iter %% 1000 ==0 )

obs_fig <- ggplot() + 
  geom_path(data=pathsdiff, aes(x=time,y=diff, group=iter,colour=iter),size=0.4)+
  geom_point(data=obs, aes(x=obs_time, y=obs),size=1.2)+
  scale_colour_gradient(low = "orange", high = "grey") +
  ylab(expression(x[2]-x[3]))+xlab("time")

longpath_fig <- longpath %>% slice(seq(1,25000000,by=500)) %>%  
    ggplot() + geom_path(aes(x=time, y=`x2_minus_x3`)) +  ylab(expression(x[2]-x[3]))

############ save figs 
pdf("paths.pdf", width=7.5, height=4)
    show(paths_fig)
dev.off()

pdf("paths_withtruth.pdf", width=7.5, height=4)
show(paths_fig_withtruth)
dev.off()


pdf("params_trace.pdf", width=7.5, height=4)
  show(params_fig)
dev.off()

pdf("params_hist.pdf", width=7.5, height=4)
  show(params_fig2)
dev.off()


pdf("obs.pdf", width=7.5, height=3)
  show(obs_fig)
dev.off()

pdf("longpath.pdf", width=7.5, height=3)
  show(longpath_fig)
dev.off()





