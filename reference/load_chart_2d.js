window.onload = function loadChart() {
  
        const data = JSON.parse(localStorage.getItem('refine')).entities;

        let alreadyExists = [];
        let distinctTags = [];
        data.forEach((item,index) => {
          if(!alreadyExists.includes(item.tag)){
            alreadyExists.push(item.tag)
            distinctTags.push({tag:item.tag,entity:item.entity})
          }
        });

        const colors = [
          "#f0f921",
          "#e3685f",
          "#a92495",
          "#ff0000",
          '#5802a3',
        ];

        function x(d) {
          return d.x;
        }
        function y(d) {
          return d.y;
        }
        function radius(d) {
          return d.size;
        }
        function color(d) {
          return d.tag;
        }

        // const color = d3.scale.category20().range(colors);
        const margin = { top: 5.5, right: 19.5, bottom: 12.5, left: 39.5 };
        let clientWidth = Math.max(document.documentElement.clientWidth || 0, window.innerWidth || 0)
        const width = 1000,
          height = 500 - margin.top - margin.bottom;

        const distortionSpeed = 2.5; // speed of distortion

        const yAxisRange = data.map((d) => d.y);
        const xAxisRange = data.map((d) => d.x);
        const sizeRange = data.map((d) => d.size);
        // const entityRange = data.map((d) => d.entity);
        const tagRange = data.map((d) => d.tag);
        const colorScaleRegion = [...new Set(tagRange)];

        // chart margins
        const minSizeRange = d3.min(sizeRange), maxSizeRange = d3.max(sizeRange);
        const yScaleMinNumber = d3.min(yAxisRange), yScaleMaxNumber = d3.max(yAxisRange);
        const xScaleMinNumber = d3.min(xAxisRange), xScaleMaxNumber = d3.max(xAxisRange);
        const circleRange = 40;
        const backgroundColor = "rgb(229, 236, 246)";

        // Various scales and distortions.
        const xScale = d3
          .scaleLinear()
          .domain([xScaleMinNumber * 1.2, xScaleMaxNumber * 1.4])
          .range([0, width]);

        const yScale = d3
          .scaleLinear()
          .domain([yScaleMinNumber * 1.2, yScaleMaxNumber * 1.4])
          .range([height, 0]);

        const radiusScale = d3
          .scaleSqrt()
          .domain([0, maxSizeRange])
          .range([0, circleRange]);

        const colorScale = d3
          .scaleOrdinal(d3.schemeCategory10)
          .domain(colorScaleRegion)
          .range(colors);

        // The x & y axes.
        const xAxis = d3
          .axisBottom(xScale)
          .tickFormat(x => x.toFixed(2))
          .tickSize(-height);

        const yAxis = d3.axisLeft(yScale).tickSize(-width);

        const svgZoom = d3.zoom().scaleExtent([0.5, 1.1]).on("zoom", () => {
          d3.select('.graph')
            .attr('transform', d3.event.transform)
        })

        const legendWidth = 135
        const svg = d3
          .select("#mychart")
          .append("svg")
          .attr("width", width + margin.left + margin.right + legendWidth)
          .attr("height", height + margin.top + margin.bottom)
          .append("g")
          .classed('graph', true)
          .call(svgZoom)
          .append("g")
          .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

        d3.select('#zoomIN').on('click', () => {
          svgZoom.scaleBy(svg.transition().duration(750), 2.2);
        })

        d3.select('#zoonOut').on('click', () => {
          svgZoom.scaleBy(svg.transition().duration(750), 0.8);
        })

        // Add a background rect for mousemove.
        svg
          .append("rect")
          .style("fill", backgroundColor)
          .attr("width", width)
          .attr("height", height);

        // Add the x-axis.
        svg
          .append("g")
          .attr("class", "x axis")
          .attr("transform", "translate(0," + height + ")")
          .call(xAxis);

        // Add the y-axis.
        svg.append("g").attr("class", "y axis").call(yAxis);

        const tooltip = d3.select("#mychart")
          .append("div")
          .attr("class", "chart-tooltip")
          .style("opacity", 0)
          .style('color', '#000');

        const outerCircle = svg
          .append("g")
          .attr("class", "dots")
          .selectAll(".dot")
          .data(data)
          .enter()
          .append('g')

          const dot = outerCircle.append("circle")
          .attr("class", "dot")
          .style("fill", (d, i) => colorScale(color(d)))
          .call(position)
          .on("mousemove", (d) => {
            tooltip.transition().duration(200).style('opacity', 0.9);
            tooltip.html(d.entity)
            .style('left', `${d3.event.layerX + 20}px`)
            .style('top', `${(d3.event.layerY)}px`)
            .style('width', 'fit-content');
            tooltip.style('background-color', colorScale(color(d)));
          })
          .on("mouseout", (d) => tooltip.transition().duration(200).style('opacity', 0))
          .sort((a, b) => radius(b) - radius(a));

        // Add label above circle
        outerCircle.append('text')
          .attr('x', (d) => xScale(x(d)))
          .attr('y', (d) => yScale(y(d)) - radiusScale(radius(d)) - 10)
          .style('font-size', '14px')
          .attr('text-anchor', 'middle')
          .style('font-family', 'sans-serif')
          .style('fill', '#000')
          .sort((a, b) => radius(b) - radius(a))
          .text((d) => d.entity)
          .text((d) => {
            if(clientWidth < 800) {
              //return mobile screen text
              let stringLength = d.entity.length
              return stringLength < 14 ? d.entity.substring(0, stringLength) : d.entity.substring(0, 14) + '..'
            } else {
              return d.entity
            }

          })
          // .on("mousemove", (d) => {
          //   console.log(this)
          //   console.log("mousemove")
          // })
          // .on("mouseout", (d) => {
          //   console.log("mouse out")
          // })

        // Adding legends in chart
        const legendSize = 15
        const legend = d3.select("#mychart").select("svg")
            .append("g")
            .attr("class", "legends")
            .attr("transform", "translate(" + (width + 50) + "," + (height - 210) + ")")

            legend
              .selectAll("mydots")
              .data(distinctTags)
              .enter()
              .append("rect")
              .attr("y", function(d,i){ return 100 + i*(legendSize+5)}) // 100 is where the first dot appears. 25 is the distance between dots
              .attr("width", legendSize)
              .attr("height", legendSize)
              .style("fill", function(d){ return colorScale(color(d))})
            // Adding legends in chart
            legend.selectAll("mylabels")
                  .data(distinctTags)
                  .enter()
                  .append("text")
                  .attr("x", legendSize*1.2)
                  .attr("y", function(d,i){ return 100 + i*(legendSize+5) + (legendSize/2)}) // 100 is where the first dot appears. 25 is the distance between dots
                  .style("fill", function(d){ return color(d)})
                  .text(function(d){ return d.tag})
                  .attr("text-anchor", "left")
                   .style('font-size','12px')
                  .style('font-family', 'sans-serif')
                  .style("alignment-baseline", "middle")
	              .style("alignment-baseline", "central")
                  .style("font-weight", "bold")

            // Positions the dots based on data.
        function position(dot) {
          dot
            .attr("cx", (d) => xScale(x(d)))
            .attr("cy", (d) => yScale(y(d)))
            .attr("r", (d) => radiusScale(radius(d)));
        }
        
      }