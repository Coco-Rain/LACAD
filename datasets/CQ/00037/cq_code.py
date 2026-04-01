import cadquery as cq

s = cq.Sketch().circle(0.5)
result = (
cq.Workplane()
.box(50, 20, 15)
.faces("<Y")
.vertices("<X and >Z")
.workplane(centerOption="CenterOfMass", offset=-4.5)
.center(0, -2.0)
.placeSketch(s)
.extrude(8, combine="s")
)
cq.exporters.export(result, 'GT.stl')