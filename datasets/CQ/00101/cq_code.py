import cadquery as cq

s = cq.Sketch().rect(10, 10).push([(5, 5)]).rect(10, 10, mode="s")
result = cq.Workplane("YZ").placeSketch(s).extrude(1)
cq.exporters.export(result, 'GT.stl')