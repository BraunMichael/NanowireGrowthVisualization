import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import re
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)


def makeMPLColormap(numColors, rotation, light, hue, colorMapDarkValue):
    Palette = sns.cubehelix_palette(n_colors=numColors, rot=rotation, light=light, hue=hue, dark=colorMapDarkValue, as_cmap=False)
    # mapPalette = sns.cubehelix_palette(n_colors=numColors, rot=rotation, light=light, hue=hue, dark=colorMapDarkValue, as_cmap=True)
    mapMPL = colors.LinearSegmentedColormap.from_list("test", Palette)
    return mapMPL


def lowerAndUpperLims(columnTitle, percentRangeOverdraw, dataframeList):
    colMin = min([dataFrame[columnTitle].min() for dataFrame in dataframeList])
    colMax = max([dataFrame[columnTitle].max() for dataFrame in dataframeList])
    colRange = colMax - colMin
    lowerLim = colMin - percentRangeOverdraw * colRange
    upperLim = colMax + percentRangeOverdraw * colRange
    return lowerLim, upperLim


def adjustFigAspect(fig, aspect: float = 1):
    """
    Adjust the subplot parameters so that the figure has the correct
    aspect ratio.
    """
    xsize,ysize = fig.get_size_inches()
    minsize = min(xsize,ysize)
    xlim = .4*minsize/xsize
    ylim = .4*minsize/ysize
    if aspect < 1:
        xlim *= aspect
    else:
        ylim /= aspect
    fig.subplots_adjust(left=.5-xlim,
                        right=.5+xlim,
                        bottom=.5-ylim,
                        top=.5+ylim)


def plotVisualization(dataframeList, yDataHeaderList, yTitlesList, xDataHeader, xTitle, circleSizeColumn, sizeLegendList, circleSizeTitle, colorColumn, colorTitle, colorMapList, colorbarMax=None, showFig=False, saveFig=True, saveName='Visualization', saveFormat='svg', yLimList=None):
    assert len(yDataHeaderList) == len(yTitlesList), "You must have a title for each header"
    assert len(dataframeList) == len(colorMapList), "You must have a colormap for each dataframe"
    if colorbarMax is None:
        _, colorbarMax = lowerAndUpperLims(colorColumn, 0, dataframeList)
    circleScaleFactor = 500
    plotAlpha = 0.7
    percentRangeOverdraw = 0.2  # add 20% of range on each side for circle's to fit nicely

    fig, axs = plt.subplots(nrows=len(yDataHeaderList), ncols=len(dataframeList), sharey='row', sharex='col', figsize=(9, 30))
    fig.set_figwidth(20)
    # fig.set_figheight(30)
    fig.set_figheight(6*len(yDataHeaderList))
    fig.subplots_adjust(wspace=0)
    fig.subplots_adjust(hspace=0)
    fig.add_subplot(111, frameon=False, facecolor='white')

    # Plot everything
    aPlot = None
    for rowIndex, (column, title) in enumerate(zip(yDataHeaderList, yTitlesList)):
        for colIndex, (dataframe, colorMap) in enumerate(zip(dataframeList, colorMapList)):
            aPlot = axs[rowIndex][colIndex].scatter(dataframe[xDataHeader], dataframe[column], s=circleScaleFactor * dataframe[circleSizeColumn], c=dataframe[colorColumn], cmap=colorMap, edgecolors='black', vmin=0, vmax=colorbarMax, alpha=plotAlpha)
            if colIndex == 0:
                axs[rowIndex][colIndex].set_ylim(lowerAndUpperLims(column, percentRangeOverdraw, dataframeList))
                axs[rowIndex][colIndex].set_ylabel(title)

    lowerXLim, upperXLim = lowerAndUpperLims(xDataHeader, percentRangeOverdraw, dataframeList)
    for axList in axs:
        for ax in axList:
            ax.set_xlim(lowerXLim + 0.01, upperXLim - 0.01)
            ax.minorticks_on()
            ax.tick_params(which='both', axis='both', direction='in', top=True, bottom=True, left=True, right=True)
            ax.tick_params(which='major', axis='both', direction='in', length=8, width=1)
            ax.tick_params(which='minor', axis='both', direction='in', length=4, width=1)
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax.yaxis.set_minor_locator(AutoMinorLocator(1))

    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel(xTitle)

    fig.subplots_adjust(right=0.8)

    # [horizontal position, vertical position, relative width, relative height]
    colorbar_ax = fig.add_axes([0.83, 0.225, 0.03, 0.3])
    cbar = fig.colorbar(aPlot, cax=colorbar_ax)
    cbar.set_alpha(1)
    cbar.draw_all()
    cbar.ax.set_ylabel(colorTitle)

    # [horizontal position, vertical position, relative width, relative height]
    legend_ax = fig.add_axes([0.85, 0.525, 0.03, 0.3])
    legend_ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    legend_ax.axis('off')
    hiddenLegendHandles = []
    hiddenLegendLabels = []
    for circleSize in sizeLegendList:
        hiddenLegendPlot = plt.scatter([], [], c='k', alpha=0.8, s=circleScaleFactor * circleSize)
        tempHandle, tempLabel = hiddenLegendPlot.legend_elements(prop="sizes")
        hiddenLegendHandles.extend(tempHandle)
        hiddenLegendLabels.extend(tempLabel)

    legend_ax.legend(hiddenLegendHandles, sizeLegendList, title=circleSizeTitle, numpoints=1, scatterpoints=5, frameon=False, labelspacing=3.5, handletextpad=2, borderaxespad=0, loc='center', borderpad=2)
    if saveFig:
        fig.savefig(saveName + '.' + saveFormat, facecolor='white', edgecolor='none', format=saveFormat)
    if showFig:
        # adjustFigAspect(fig, aspect=.3)
        plt.show()


SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 22

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


df = pd.read_csv('./LatestDataset.csv')
# SnSurfaceStr = 'Sn Surface Coverage'
SnSurfaceStr = 'Sn Surface Coverage (percent total area)'

df = df.dropna(subset=[SnSurfaceStr])
df = df.apply(pd.to_numeric, errors='ignore')

# Sort the values by XRD Sn Content in descending order so the smallest Sn content values (circles) get drawn last (on top of everything else)
# Ties broken by Sn Surface Coverage
df_resorted = df.sort_values(by=['XRD Sn Content', SnSurfaceStr], kind='mergesort', ascending=False)

# Have a dataframe with numerical values for position if needed (ie some plots can't plot categorical data)
# df_resorted_sub = df_resorted.replace({'Center': 0, 'Andrew': 1, 'Reflectometry': 2}, inplace=False)

colorMapDarkValue = 0.15  # (0 makes highest value black, going to 1)
RedMapMPL = makeMPLColormap(numColors=30, rotation=0.5, light=1, hue=1, colorMapDarkValue=colorMapDarkValue)
GreenMapMPL = makeMPLColormap(numColors=30, rotation=-0.5, light=1, hue=1, colorMapDarkValue=colorMapDarkValue)
BlueMapMPL = makeMPLColormap(numColors=30, rotation=-0.1, light=1, hue=1, colorMapDarkValue=colorMapDarkValue)

df_Center = df_resorted.loc[df_resorted['Position'] == 'Center']
df_Reflectometry = df_resorted.loc[df_resorted['Position'] == 'Reflectometry']
df_Andrew = df_resorted.loc[df_resorted['Position'] == 'Andrew']

dataframeList = [df_Center, df_Andrew, df_Reflectometry]
colorMapList = [RedMapMPL, RedMapMPL, RedMapMPL]

yDataHeaderList = ['NV', 'Germane Flow', 'Wire Density']
yTitlesList = ['Needle Valve', 'Germane Flow (sccm)', 'Nanowire Density (μm$^{-2}$)']
xDataHeader = 'Temperature'
xTitle = 'Temperature (°C)'
circleSizeColumn = 'XRD Sn Content'
circleSizeTitle = 'XRD Sn Content'
sizeLegendList = [4, 8, 12]  # [4, 8, 12] for Sn content as circle sizes

colorColumn = SnSurfaceStr
colorTitle = 'Surface Sn Coverage Percent'

# Plot all data
plotVisualization(dataframeList, yDataHeaderList, yTitlesList, xDataHeader, xTitle, circleSizeColumn, sizeLegendList, circleSizeTitle, colorColumn, colorTitle, colorMapList, saveName='GeSnVisualization_inclRandomXRD')

# Only for actually measured XRD
dataframeList = [dataframe.dropna(subset=['XRD Measured']) for dataframe in dataframeList]
plotVisualization(dataframeList, yDataHeaderList, yTitlesList, xDataHeader, xTitle, circleSizeColumn, sizeLegendList, circleSizeTitle, colorColumn, colorTitle, colorMapList, saveName='GeSnVisualization_measXRD')

# Only for actually measured XRD and Wire params
dataframeList = [dataframe.dropna(subset=['Average Wire Width (nm)']) for dataframe in dataframeList]
yDataHeaderList = ['NV', 'Germane Flow', 'Wire Density', 'Average Wire Width (nm)', 'Average Wire Length (nm)']
yTitlesList = ['Needle Valve', 'Germane Flow (sccm)', 'Nanowire Density (μm$^{-2}$)', 'Nanowire Width (nm)', 'Nanowire Length (nm)']
plotVisualization(dataframeList, yDataHeaderList, yTitlesList, xDataHeader, xTitle, circleSizeColumn, sizeLegendList, circleSizeTitle, colorColumn, colorTitle, colorMapList, saveName='GeSnVisualization_measXRDandWireParams', showFig=True)