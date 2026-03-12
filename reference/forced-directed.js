function loadForChart() {
        const data = JSON.parse(localStorage.getItem('refine'));
        let groupData = _.chain(data.entities).groupBy("cluster").value();
        const nodes = [];
        const links = [];
        let sourceCounter = 1;
        let targetCounter = 1;
        let changedGroupValue = 1;
        let oldKey = "";
        let id = 0;
        Object.keys(groupData).forEach(function (item, key) {
          let groupList = groupData[item];
          groupList.forEach(function (innerGroupItem, innerKey) {
            id++;
            nodes.push({
              id: id,
              label: innerGroupItem.entity,
              group: innerGroupItem.cluster,
              size: innerGroupItem.size === 0 ? 0 : Math.log(innerGroupItem.size, 10),
            });

            if (oldKey !== "") {
              links.push({
                source: changedGroupValue,
                target: sourceCounter,
                value: 1,
              });
              targetCounter = sourceCounter;
              // sourceCounter += 1
              oldKey = "";
            }

            links.push({
              source: sourceCounter,
              target: targetCounter,
              value: 1,
            });
            sourceCounter += 1;
          });
          // storing parent loop key
          oldKey = key;
          // targetCounter = sourceCounter
          changedGroupValue = targetCounter;
        });
        links.shift();
        document.getElementById('chart1').innerHTML = null;
        const chartElement = document.getElementById('chart1')
        const divSize = chartElement.getBoundingClientRect()
        const width = divSize.width,
          height = divSize.height;

        const colors = [
          "#857c85",
          "#d27d7d",
          "#c16548",
          "#b960a7",
          "#ddbd4d",
          "#705177",
        ];

        var fisheye = d3.fisheye.circular().radius(200);

        const zoom = d3.zoom().scaleExtent([0.5, 2]).on("zoom", zoomed);

        function zoomed() {
          node.attr("transform", d3.event.transform);
          link.attr("transform", d3.event.transform);
          text.attr("transform", d3.event.transform);
        }

        var svg = d3
            .select("#chart1")
            .append("svg")
            .attr("width", width)
            .attr("height", height + 50),
          transform = d3.zoomIdentity;

        // custom color range
        // var color = d3.scaleOrdinal(d3.schemeCategory20).range(colors);

        // default d3 color range
        const color = d3.scaleOrdinal(d3.schemeCategory20)
        svg.call(zoom);

        d3.select("#zoomIN").on("click", () => {
          zoom.scaleBy(svg.transition().duration(750), 1.2);
        });

        d3.select("#zoonOut").on("click", () => {
          zoom.scaleBy(svg.transition().duration(750), 0.8);
        });

        var simulation = d3
          .forceSimulation()
          .force(
            "link",
            d3.forceLink().id(function (d) {
              return d.id;
            }).distance(15).strength(0.13)
          )
          .force("charge", d3.forceManyBody())
          .force("center", d3.forceCenter(width / 2, height / 2));

        simulation.nodes(nodes).on("tick", ticked);

        simulation.force("link").links(links);

        const tooltip = d3
          .select("#tooltip")
          .style("opacity", 0)
          .style("color", "#000");

        var link = svg
          .append("g")
          .attr("class", "links")
          .selectAll("line")
          .data(links)
          .enter()
          .append("line")
          .attr("stroke-width", function (d) {
            return Math.sqrt(d.value);
          });

        const nodeGroup = svg.append("g").attr("class", "nodes").selectAll("node");

        const singlNodeGroup = svg
          .selectAll(".node")
          .data(nodes)
          .enter()
          .append("g")
          // .attr("transform", (d) => "translate(" + d.x + "," + d.y + ")")

        var node = singlNodeGroup
          .append("circle")
          .classed("node", true)
          .attr("r", 5)
          .attr("cx", function (d) {
            return d.x;
          })
          .attr("cy", function (d) {
            return d.y;
          })
          .attr("fill", function (d) {
            return color(d.group);
          })
          .call(
            d3
              .drag()
              .on("start", dragstarted)
              .on("drag", dragged)
              .on("end", dragended)
          )
          .on("mouseover", (d) => {
            tooltip.transition().duration(200).style("opacity", 0.9);
            tooltip
              .html(d.label)
              .style("left", `${d3.event.layerX + 20}px`)
              .style("top", `${d3.event.layerY}px`)
              .style("width", "fit-content");
            tooltip.style("background-color", color(d.group));
          })
          .on("mouseout", function (d) {
            tooltip.transition().duration(200).style("opacity", 0);
          });

        var text = singlNodeGroup
          .append("text")
          .attr("class", "text")
          .attr("y", function(d) {
            return d.y;
          })
          .attr("x", function(d) {
            return d.x + 10;
          })
          .text(function(d) {
            return d.label;
          });

        function ticked() {
          link
            .attr("x1", function (d) {
              return d.source.x;
            })
            .attr("y1", function (d) {
              return d.source.y;
            })
            .attr("x2", function (d) {
              return d.target.x;
            })
            .attr("y2", function (d) {
              return d.target.y;
            });

          node
            .attr("cx", function (d) {
              return d.x;
            })
            .attr("cy", function (d) {
              return d.y;
            });

          text.attr("x", function (d) {
              return d.x + 10;
            })
            .attr("y", function (d) {
              return d.y + 5;
            });
        }

        svg.on("mousemove", function () {
          fisheye.focus(d3.mouse(this));
          node
            .each(function (d) {
              d.fisheye = fisheye(d);
            })
            .attr("cx", function (d) {
              return d.fisheye.x;
            })
            .attr("cy", function (d) {
              return d.fisheye.y;
            })
            .attr("r", function (d) {
              return d.fisheye.z * 4.5;
            });

          link
            .attr("x1", function (d) {
              return d.source.fisheye.x;
            })
            .attr("y1", function (d) {
              return d.source.fisheye.y;
            })
            .attr("x2", function (d) {
              return d.target.fisheye.x;
            })
            .attr("y2", function (d) {
              return d.target.fisheye.y;
            });

          singlNodeGroup
            .select("text")
            .attr("x", (d) => {
                return d.fisheye.x + 10;
            })
            .attr("y", function(d) {
                return d.fisheye.y + 2;
            });
        });

        function dragstarted(d) {
          if (!d3.event.active) simulation.alphaTarget(0.3).restart();
          d.fx = d.x;
          d.fy = d.y;
        }

        function dragged(d) {
          d.fx = d3.event.x;
          d.fy = d3.event.y;
        }

        function dragended(d) {
          if (!d3.event.active) simulation.alphaTarget(0);
          d.fx = d.x;
          d.fy = d.y;
        }
      }

      function redraw() {
        window.location.reload()
      }
loadForChart();