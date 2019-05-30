# To create the animated DBSCAN alliander logo


# Creates a modification to the pattern generation code in:
# https://www.naftaliharris.com/blog/visualizing-dbscan-clustering/

from PIL import Image
import numpy as np


im = Image.open('alliander.png')
rgb_im = im.convert('RGB')

w, h = im.width, im.height


N = 10000

s = []


scale = 25 / w

for i in range(N):
    x, y = int(np.random.uniform(w)), int(np.random.uniform(h))
    if rgb_im.getpixel((x, y))[0] < 127:
        a,b = np.random.normal(0,.05,2)

        x, y = x+a, y+b

        x = (x - w/2) * scale
        y = -(y - h/2) * scale

        s.append("{" + "x: {0}, y: {1}, cluster: 0".format(x+a, y+b) + "}")

# Overrides the draw function to use smaller dots
output = """draw = function (data) {
    var points = svg.selectAll(".dot")
      .data(data)
    .enter().append("circle")
      .attr("class", "dot")
      .attr("r", 1)
      .attr("cx", function(d, i) { return x(30 * Math.cos(i / 5)); })
      .attr("cy", function(d, i) { return y(30 * Math.sin(i / 5)); })
      .style("fill", function(d) { return color(d.cluster); })
      .style("stroke", "black")
      .style("stroke-width", "1px");

    points.transition()
    .duration(500)
    .attr("cx", function(d) { return x(d.x); })
    .attr("cy", function(d) { return y(d.y); });
}\n\n"""

# Overrides the smiley pattern generation code
output += "smiley=function(){return ["+",\n".join(s)+"]}\n\n"

print(output)
